import os
import glob
import datetime
import torch
import numpy as np
import torch.nn as nn
from rdkit import Chem
from utils.data_utils import ReactionDataset, BEmatrix_to_mol, ps
from utils.rounding import saferound_tensor
import torch.distributed as dist
from train import init_model, init_loader
from utils.train_utils import log_rank_0, setup_logger, log_args
from settings import Args
from collections import defaultdict
import time

def is_sym(a):
    return (a.transpose(1, 0) == a).all()

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

def standardize_smiles(mol):
    if mol is None: return "None"
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)

def tau_leaping_batch(model, y_emb, y_len, x0, steps=100, device='cuda'):
    """
    Joint factorized Tau-Leaping. 
    Predicts electron jumps while respecting molecular boundaries.
    """
    B, N, _ = x0.shape
    xt = x0.clone()
    dt = 1.0 / steps
    mask = y_len_to_mask(y_len, N)
    
    time_grid = torch.linspace(0, 1, steps + 1, device=device)

    for i in range(steps):
        t = torch.full((B,), time_grid[i], device=device)
        
        s_logits, t_logits = model(y_emb, y_len, xt, t)

        s_logits = 0.5 * (s_logits + s_logits.transpose(1, 2))
        t_logits = 0.5 * (t_logits + t_logits.transpose(1, 2))
        
        s_flat = s_logits[..., 1].view(B, -1).masked_fill(mask, -1e12)
        t_flat = t_logits[..., 1].view(B, -1).masked_fill(mask, -1e12)

        sum_s = torch.logsumexp(s_flat, dim=-1).exp()
        sum_t = torch.logsumexp(t_flat, dim=-1).exp()
        global_rate = sum_s * sum_t
        
        total_jumps = torch.poisson(global_rate * dt)
        s_probs = torch.softmax(s_flat, dim=-1)
        t_probs = torch.softmax(t_flat, dim=-1)

        for b in range(B):
            n_jumps = int(total_jumps[b].item())
            if n_jumps == 0: continue

            src_idx = torch.multinomial(s_probs[b], n_jumps, replacement=True)
            snk_idx = torch.multinomial(t_probs[b], n_jumps, replacement=True)

            for j in range(n_jumps):
                s_u, s_v = divmod(int(src_idx[j]), N)
                k_u, k_v = divmod(int(snk_idx[j]), N)

                xt[b, s_u, s_v] -= 1
                if s_u != s_v: xt[b, s_v, s_u] -= 1
                xt[b, k_u, k_v] += 1
                if k_u != k_v: xt[b, k_v, k_u] += 1

    return xt

def get_predictions(args, model, flow, data_loader, iter_count=np.inf, write_o=None):
    accuracy = []
    model.eval()
    
    with torch.no_grad():
        log_rank_0('Start Tau-Leaping Evaluation...')
        inferenced_indexes = set()

        for batch_idx, data_batch in enumerate(data_loader):
            if batch_idx >= iter_count: break
            data_batch.to(args.device)

            x0 = data_batch.src_matrices
            y_len = data_batch.src_lens
            y_emb = model.id2emb(data_batch.src_token_ids)
            B, N, _ = x0.shape

            sample_size = getattr(args, 'sample_size', 1)
            y_emb_rep = y_emb.repeat_interleave(sample_size, dim=0)
            y_len_rep = y_len.repeat_interleave(sample_size, dim=0)
            x0_rep = x0.repeat_interleave(sample_size, dim=0)

            xt_final = tau_leaping_batch(model, y_emb_rep, y_len_rep, x0_rep, 
                                        steps=getattr(args, 'inference_steps', 100), 
                                        device=args.device)

            if dist.is_initialized():
                res = (data_batch.src_data_indices, xt_final.cpu(), x0.cpu(), 
                       y_len.cpu(), data_batch.src_smiles_list, data_batch.tgt_smiles_list)
                gathered = [None] * dist.get_world_size()
                dist.all_gather_object(gathered, res)
            else:
                gathered = [(data_batch.src_data_indices, xt_final.cpu(), x0.cpu(), 
                            y_len.cpu(), data_batch.src_smiles_list, data_batch.tgt_smiles_list)]

            if dist.get_rank() != 0: continue

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
                            if standardize_smiles(pred_mol) == gold_smi:
                                correct_found = True
                                break
                        except: continue
                    
                    accuracy.append([1 if correct_found else 0])
                    if write_o: write_o.write(f"{d_idx}|{1 if correct_found else 0}\n")

    return accuracy
