import os
import glob
import datetime
import torch
import numpy as np
import torch.nn as nn
from rdkit import Chem, RDLogger
from utils.data_utils import ReactionDataset, BEmatrix_to_mol, ps
from utils.rounding import saferound_tensor
import torch.distributed as dist
from utils.train_utils import log_rank_0
from settings import Args
from collections import defaultdict
import time

# Globally disable RDKit logs for clean execution
RDLogger.DisableLog('rdApp.*')

def is_sym(a):
    return (a.transpose(1, 0) == a).all()

def standardize_smiles(mol):
    if mol is None: return "None"
    
    # 1. Remove Atom Mapping
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    
    # 2. Sanitize
    try:
        Chem.SanitizeMol(mol)
    except:
        pass
        
    # 3. Canonicalize
    try:
        mol = Chem.RemoveHs(mol)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except:
        pass

    # Fallback
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)

def y_len_to_mask(y_len, N):
    """Creates a (B, N*N) mask where padding is True."""
    B = y_len.shape[0]
    node_range = torch.arange(N, device=y_len.device).unsqueeze(0)
    node_mask = node_range >= y_len.unsqueeze(1) 
    matrix_mask = node_mask.unsqueeze(1) | node_mask.unsqueeze(2)
    return matrix_mask.view(B, -1)

def redist_fix(pred_matrix, reac_smi, reac_be_matrix):
    """Ensures atom-level electron conservation via lone-pair adjustment."""
    pred_electron_sum = np.sum(pred_matrix, axis=1) + np.sum(pred_matrix, axis=0) - np.diag(pred_matrix)
    reac_electron_sum = np.sum(reac_be_matrix, axis=1) + np.sum(reac_be_matrix, axis=0) - np.diag(reac_be_matrix)
    diff = reac_electron_sum - pred_electron_sum
    if np.isclose(np.sum(diff), 0, atol=1e-5):
        diag_idx = np.diag_indices_from(pred_matrix)
        pred_matrix[diag_idx] += diff
    return pred_matrix

def custom_round(x, target_sums=None):
    output = []
    for i in range(x.shape[0]):
        t_sum = target_sums[i].item() if target_sums is not None else x[i].sum().item()

        current_sum = x[i].sum()
        if current_sum > 0:
            norm_x = x[i] * (t_sum / current_sum)
        else:
            norm_x = x[i]

        try:
            rounded = saferound_tensor(norm_x, places=0, strategy="difference", topline=t_sum)
            output.append(rounded)
        except:
            output.append(torch.round(norm_x))
            
    return torch.stack(output).clamp(min=0)

def tau_leaping_batch_scatter(
    model,
    y_emb,
    y_len,
    x0,
    steps=100,
    device="cuda",
    max_jumps_per_atom=2,
    rate_scalar=6.0,
):
    B, N, _ = x0.shape
    xt = x0.clone()
    dt = 1.0 / steps
    mask = y_len_to_mask(y_len, N)
    stride = N * N

    for i in range(steps):
        t = torch.full((B,), i / steps, device=device)
        
        s_logits, t_logits, stop_logits = model(y_emb, y_len, xt, t)

        # --- SOURCE ---
        s_flat = (0.5 * (s_logits + s_logits.transpose(1, 2)))[..., 1].view(B, -1)
        s_flat = s_flat.masked_fill(mask, -1e12)
        
        log_sum_s = torch.logsumexp(s_flat, dim=-1)       
        s_stop_logit = stop_logits[:, 0]                  
        log_total_s = torch.logaddexp(log_sum_s, s_stop_logit)
        p_active_s = torch.exp(log_sum_s - log_total_s)

        # --- SINK ---
        t_flat = (0.5 * (t_logits + t_logits.transpose(1, 2)))[..., 1].view(B, -1)
        t_flat = t_flat.masked_fill(mask, -1e12)
        
        log_sum_t = torch.logsumexp(t_flat, dim=-1)
        t_stop_logit = stop_logits[:, 1]
        log_total_t = torch.logaddexp(log_sum_t, t_stop_logit)
        p_active_t = torch.exp(log_sum_t - log_total_t)
        
        # --- RATE CALCULATION ---
        base_rate = np.sqrt(rate_scalar) 
        rates = (base_rate * p_active_s) * (base_rate * p_active_t)

        # --- Standard Poisson Step ---
        max_j = max_jumps_per_atom * y_len
        total_jumps = torch.minimum(torch.poisson(rates * dt).long(), max_j)

        if total_jumps.max() == 0:
            continue

        max_batch_j = total_jumps.max().item()
        
        src_samples = torch.multinomial(torch.softmax(s_flat, dim=-1), max_batch_j, replacement=True)
        tgt_samples = torch.multinomial(torch.softmax(t_flat, dim=-1), max_batch_j, replacement=True)

        jump_mask = torch.arange(max_batch_j, device=device).unsqueeze(0) < total_jumps.unsqueeze(1)
        
        src_idx = src_samples[jump_mask]
        tgt_idx = tgt_samples[jump_mask]
        batch_ids = torch.arange(B, device=device).repeat_interleave(total_jumps)

        global_src_idx = (batch_ids * stride) + src_idx
        global_tgt_idx = (batch_ids * stride) + tgt_idx

        flat_src_counts = torch.zeros(B * stride, device=device)
        flat_tgt_counts = torch.zeros(B * stride, device=device)

        flat_src_counts.scatter_add_(0, global_src_idx, torch.ones_like(src_idx, dtype=torch.float))
        flat_tgt_counts.scatter_add_(0, global_tgt_idx, torch.ones_like(tgt_idx, dtype=torch.float))

        src_counts = flat_src_counts.view(B, -1)
        tgt_counts = flat_tgt_counts.view(B, -1)
        
        flat_xt = xt.view(B, -1)
        
        # --- UPDATE & CONSERVATION CHECK ---
        real_src_counts = torch.minimum(src_counts, flat_xt)
        flat_xt = flat_xt - real_src_counts + tgt_counts

        xt = flat_xt.view(B, N, N)
        xt = 0.5 * (xt + xt.transpose(1, 2))
        xt = torch.clamp(xt, min = 0.0)

    return xt

def get_predictions(args, model, flow, data_loader, iter_count=np.inf, write_o=None):
    accuracy = []
    model.eval()
    
    with torch.no_grad():
        inferenced_indexes = set()

        for batch_idx, data_batch in enumerate(data_loader):
            if batch_idx >= iter_count: break
            data_batch.to(args.device)

            x0 = data_batch.src_matrices
            
            # Sanitization: Replace padding (-30) with 0.0 for physics calcs
            x0 = torch.where(x0 < -1, torch.tensor(0.0, device=x0.device), x0)

            y_len = data_batch.src_lens
            if hasattr(model, "module"):
                y_emb = model.module.id2emb(data_batch.src_token_ids)
            else:
                y_emb = model.id2emb(data_batch.src_token_ids)
            
            B, N, _ = x0.shape

            sample_size = getattr(args, 'sample_size', 1)
            y_emb_rep = y_emb.repeat_interleave(sample_size, dim=0)
            y_len_rep = y_len.repeat_interleave(sample_size, dim=0)
            x0_rep = x0.repeat_interleave(sample_size, dim=0)

            # --- GPU INFERENCE ---
            rate_scalar = getattr(args, 'rate_scalar', 6.0)
            xt_final = tau_leaping_batch_scatter(
                model,
                y_emb_rep,
                y_len_rep,
                x0_rep,
                steps=getattr(args, 'inference_steps', 100),
                device=args.device,
                max_jumps_per_atom=getattr(args, "max_jumps_per_atom", 2),
                rate_scalar=rate_scalar
            )
            
            # --- CPU TRANSFER ---
            xt_final = xt_final.cpu()
            x0_rep_cpu = x0_rep.cpu()
            true_sums = x0_rep_cpu.sum(dim=(1, 2))
            
            # --- ROUNDING ---
            xt_rounded = custom_round(xt_final, target_sums=true_sums)
            
            # --- GATHER ---
            if dist.is_initialized():
                res = (data_batch.src_data_indices, xt_rounded, x0.cpu(), 
                       y_len.cpu(), data_batch.src_smiles_list, data_batch.tgt_smiles_list)
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, res)
            else:
                gathered = [(data_batch.src_data_indices, xt_rounded, x0.cpu(), 
                            y_len.cpu(), data_batch.src_smiles_list, data_batch.tgt_smiles_list)]

            if dist.is_initialized() and dist.get_rank() != 0: continue

            # --- RDKIT VALIDATION ---
            for batch_res in gathered:
                indices, xt_list, x0_list, lens, src_smis, tgt_smis = batch_res
                
                for b in range(len(indices)):
                    d_idx = int(indices[b])
                    if d_idx in inferenced_indexes: continue
                    inferenced_indexes.add(d_idx)

                    correct_found = False
                    reac_mol = Chem.MolFromSmiles(src_smis[b], ps)
                    gold_smi = standardize_smiles(Chem.MolFromSmiles(tgt_smis[b], ps))
                    
                    samples = xt_list[b*sample_size : (b+1)*sample_size]
                    
                    for s_idx in range(sample_size):
                        pred_be = samples[s_idx][:lens[b], :lens[b]].numpy()
                        reac_be = x0_list[b][:lens[b], :lens[b]].numpy()
                        pred_be = redist_fix(pred_be, src_smis[b], reac_be)
                        
                        try:
                            pred_mol = BEmatrix_to_mol(reac_mol, pred_be)
                            Chem.SanitizeMol(pred_mol) 
                            pred_smi = standardize_smiles(pred_mol)
                            
                            if pred_smi == gold_smi:
                                correct_found = True
                                break
                        except Exception: 
                            continue
                    
                    accuracy.append([1 if correct_found else 0])
                    if write_o: write_o.write(f"{d_idx}|{1 if correct_found else 0}\n")

    return accuracy
