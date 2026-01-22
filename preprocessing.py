import os
import argparse
import torch
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import sys
from functools import partial

from utils.data_utils import get_BE_matrix, smi2vocabid, process_arrows, MATRIX_PAD
from utils.arrow_pushing import get_arrow_pushing

def process_one_reaction(line, max_atoms=150):
    """
    Worker function to process a single line of text.
    Returns a dictionary of tensors/data or None if failure or too large.
    """
    line = line.strip()
    if not line or ">>" not in line:
        return None

    try:
        src_smi, tgt_smi = line.split('|')[0].split('>>')
        
        if len(src_smi) > (max_atoms * 4): 
            return None 

        src_vocab_id_list, src_len = smi2vocabid(src_smi)
        
        if src_len > max_atoms:
            return None
        
        tgt_vocab_id_list, tgt_len = smi2vocabid(tgt_smi)
        
        src_matrix = get_BE_matrix(src_smi)
        tgt_matrix = get_BE_matrix(tgt_smi)
        
        delta_matrix = tgt_matrix - src_matrix
        raw_arrows = get_arrow_pushing(delta_matrix)
        arrow_tensor = process_arrows(raw_arrows)

        # Basic Checks
        assert (src_vocab_id_list == tgt_vocab_id_list).all()
        assert src_len == tgt_len

        return {
            'src_smi': src_smi,
            'tgt_smi': tgt_smi,
            'src_token_ids': src_vocab_id_list,
            'tgt_token_ids': tgt_vocab_id_list,
            'src_len': src_len,
            'tgt_len': tgt_len,
            'src_matrix': src_matrix,
            'tgt_matrix': tgt_matrix,
            'src_arrows': arrow_tensor
        }

    except Exception as e:
        return None

def preprocess_dataset(input_path, output_path, num_workers=16, max_atoms=150):
    print(f"Reading from {input_path}...")
    with open(input_path, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} reactions with {num_workers} workers (Max Atoms: {max_atoms})...")
    
    data_storage = {
        'src_smis': [],
        'tgt_smis': [],
        'src_token_ids': [],
        'tgt_token_ids': [],
        'src_lens': [],
        'tgt_lens': [],
        'src_matrices': [],
        'tgt_matrices': [],
        'src_arrows': []
    }

    worker_func = partial(process_one_reaction, max_atoms=max_atoms)

    with Pool(num_workers) as p:
        results = list(tqdm(p.imap(worker_func, lines), total=len(lines)))

    print("Aggregating results...")
    valid_count = 0
    skipped_count = 0
    
    for res in results:
        if res is None:
            skipped_count += 1
            continue
            
        valid_count += 1
        data_storage['src_smis'].append(res['src_smi'])
        data_storage['tgt_smis'].append(res['tgt_smi'])
        data_storage['src_token_ids'].append(res['src_token_ids']) 
        data_storage['tgt_token_ids'].append(res['tgt_token_ids'])
        data_storage['src_lens'].append(res['src_len'])
        data_storage['tgt_lens'].append(res['tgt_len'])

        data_storage['src_matrices'].append(torch.from_numpy(res['src_matrix']).float())
        data_storage['tgt_matrices'].append(torch.from_numpy(res['tgt_matrix']).float())
        
        data_storage['src_arrows'].append(torch.from_numpy(res['src_arrows']).float())

    print(f"Successfully processed {valid_count}/{len(lines)} reactions.")
    print(f"Skipped {skipped_count} reactions (due to errors or > {max_atoms} atoms).")
    
    print(f"Saving to {output_path}...")
    torch.save(data_storage, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Path to input .txt file")
    parser.add_argument('--output', type=str, required=True, help="Path to output .pt file")
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max_atoms', type=int, default=150, help="Filter out molecules larger than this") # <--- ADDED
    args = parser.parse_args()

    preprocess_dataset(args.input, args.output, args.workers, args.max_atoms)
