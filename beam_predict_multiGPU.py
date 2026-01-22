import os
import torch
import numpy as np
from rdkit import Chem
from utils.data_utils import ReactionDataset, BEmatrix_to_mol, ps
import torch.distributed as dist
from train import init_model, init_loader
from utils.train_utils import log_rank_0, setup_logger, log_args
from eval_multiGPU import custom_round, tau_leaping_batch, redist_fix
from settings import Args
from collections import defaultdict
import networkx as nx
import pickle
import torch.multiprocessing as mp
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def standardize_smiles(mol):
    if mol is None: return "None"
    [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)

def select(args, frontiers_dict, graph_list):
    filtered_frontiers_dict = {}
    for g_idx, frontiers in frontiers_dict.items():
        graph, root, _ = graph_list[g_idx]
        rank_frontiers = {}
        for frontier in frontiers:
            probs_list = []
            for path in nx.all_simple_paths(graph, root, frontier):
                edges = list(nx.utils.pairwise(path))
                probs = [graph.get_edge_data(u, v)['count'] / args.sample_size for u, v in edges]
                probs_list.append(np.prod(probs))
            
            rank_frontiers[frontier] = -max(probs_list) if probs_list else 0
        
        sorted_frontiers = sorted(rank_frontiers.items(), key=lambda x: x[1])[:args.beam_size]
        filtered_frontiers_dict[g_idx] = [f[0] for f in sorted_frontiers]
    return filtered_frontiers_dict

def expand(args, model, flow, data_loader):
    sample_size = args.sample_size
    overall_dict = {}

    for batch_idx, data_batch in enumerate(data_loader):
        data_batch.to(args.device)
        y = data_batch.src_token_ids
        y_len = data_batch.src_lens
        x0 = data_batch.src_matrices
        src_smis = data_batch.src_smiles_list
        B, N, _ = x0.shape

        y_emb_rep = model.id2emb(y).repeat_interleave(sample_size, dim=0)
        y_len_rep = y_len.repeat_interleave(sample_size, dim=0)
        x0_rep = x0.repeat_interleave(sample_size, dim=0)

        final_states = tau_leaping_batch(
            model, y_emb_rep, y_len_rep, x0_rep, 
            steps=getattr(args, 'inference_steps', 100), device=args.device
        )
        
        rounded_states = custom_round(final_states)
        states_per_mol = torch.split(rounded_states, sample_size)

        for b_idx in range(B):
            reac_smi = src_smis[b_idx]
            reac_mol = Chem.MolFromSmiles(reac_smi, ps)
            mol_samples = states_per_mol[b_idx]
            
            matrices, counts = torch.unique(mol_samples, dim=0, return_counts=True)
            matrices, counts = matrices.cpu().numpy(), counts.cpu().numpy()
            
            pred_smis_dict = defaultdict(int)
            num_nodes = y_len[b_idx].item()
            reac_be = x0[b_idx][:num_nodes, :num_nodes].cpu().numpy()

            for i in range(matrices.shape[0]):
                pred_be = matrices[i][:num_nodes, :num_nodes]
                if abs(pred_be.sum() - reac_be.sum()) > 1e-3: continue

                try:
                    pred_be = redist_fix(pred_be, reac_smi, reac_be)
                    pred_mol = BEmatrix_to_mol(reac_mol, pred_be)
                    smi = standardize_smiles(pred_mol)
                    if smi != "None":
                        pred_smis_dict[smi] += counts[i]
                except: continue

            top_preds = sorted(pred_smis_dict.items(), key=lambda x: x[1], reverse=True)[:args.nbest]
            overall_dict[reac_smi] = dict(top_preds)

    return overall_dict

def reactant_process(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol, explicitOnly=False)
        for idx, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(idx+1)
        return Chem.MolToSmiles(mol, isomericSmiles=False, allHsExplicit=True)
    except: return smi

def clean(smi):
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None: return smi
    mol = Chem.RemoveHs(mol)
    [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def beam_search(args, model, flow, frontiers_dict, graph_list):
    smiles_list = [f for frontiers in frontiers_dict.values() for f in frontiers]
    if not smiles_list: return
    
    exclude_gidx = [idx for idx, (g, r, _) in enumerate(graph_list) 
                    if g.nodes[r]['depth'] >= args.max_depth]

    test_dataset = ReactionDataset(args, smiles_list, reactant_only=True)
    try:
        test_loader = init_loader(args, test_dataset, batch_size=args.test_batch_size, 
                                shuffle=False, use_sort=False)
    except: return
    
    overall_dict = expand(args, model, flow, test_loader)
    new_frontiers_dict = defaultdict(list)
    
    for g_idx, frontiers in frontiers_dict.items():
        if g_idx in exclude_gidx: continue
        graph, root, _ = graph_list[g_idx]
        for frontier in frontiers:
            if frontier not in overall_dict: continue
            for rank, (product, count) in enumerate(overall_dict[frontier].items()):
                if not graph.has_node(product):
                    graph.add_node(product)
                    new_frontiers_dict[g_idx].append(product)
                graph.add_edge(frontier, product, rank=rank, count=count)

    filtered_frontiers = select(args, new_frontiers_dict, graph_list)
    beam_search(args, model, flow, filtered_frontiers, graph_list)

def worker(rank, args, chunk, chunk_idx, lock, queue):
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    args.device = device
    
    checkpoint = os.path.join(args.model_path, args.model_name)
    state = torch.load(checkpoint, map_location=device)
    model, flow, _ = init_model(state["args"])
    sd = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
    model.load_state_dict(sd)
    model.eval()
    
    graph_list = []
    frontiers_dict = defaultdict(list)
    for idx, line in enumerate(chunk):
        parts = line.strip().split(">>")
        ori_reac = parts[0]
        targets = parts[1].split("|") if len(parts) > 1 else []
        targets = [standardize_smiles(Chem.MolFromSmiles(s)) for s in targets]
        
        reactant = reactant_process(ori_reac)
        graph = nx.DiGraph()
        graph.add_node(reactant, depth=1)
        graph_list.append((graph, reactant, (ori_reac, targets)))
        frontiers_dict[idx].append(reactant)
    
    beam_search(args, model, flow, frontiers_dict, graph_list)
    queue.put((rank, chunk_idx, graph_list))

def check_if_successful(graph, products):
    achieved = set()
    target_set = set(products)
    for node in graph.nodes():
        node_parts = set(clean(node).split('.'))
        match = node_parts & target_set
        if match: achieved.update(match)
    return achieved

def main_multi_gpu(args):
    world_size = torch.cuda.device_count()
    with open(args.test_path, 'r') as f:
        test_smiles_list = f.readlines()
    
    chunk_size = args.chunk_size // world_size
    chunks = [test_smiles_list[i:i + chunk_size] for i in range(0, len(test_smiles_list), chunk_size)]
    
    os.makedirs(args.result_path, exist_ok=True)
    lock = mp.Lock()
    q = mp.Queue()
    
    chunk_idx = 0
    for g_id, group in enumerate([chunks[i:i+world_size] for i in range(0, len(chunks), world_size)]):
        processes = []
        for gpu_idx, chunk in enumerate(group):
            p = mp.Process(target=worker, args=(gpu_idx, args, chunk, chunk_idx, lock, q))
            p.start()
            processes.append(p)
            chunk_idx += 1
        
        all_results = []
        for _ in processes:
            res = q.get()
            for b_idx, (g, r, (orig, targets)) in enumerate(res[2]):
                achieved = check_if_successful(g, targets)
                all_results.append((g, r, (orig, targets), achieved))
        
        for p in processes: p.join()
        with open(os.path.join(args.result_path, f'res_gpu_chunk_{g_id}.pickle'), "wb") as f_out:
            pickle.dump(all_results, f_out)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main_multi_gpu(Args)
