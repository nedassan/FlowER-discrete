import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import multiprocessing as mp

def process_sample_batch_gpu(batch_data, device):
    """Process a batch of samples on GPU and return SVD results"""
    results = []
    
    for b, sample in batch_data:
        try:
            src_matrix = torch.as_tensor(sample['src_matrices'])
            arrows = torch.as_tensor(sample['src_arrows'])
            
            N = src_matrix.shape[0]
            R = torch.zeros((N * N, N * N), device=device)
            
            if arrows.ndim == 1:
                arrows = arrows.unsqueeze(0)

            arrows = arrows.to(device)
            for arrow in arrows:
                if arrow[0] == -1:
                    break
                
                u_s, v_s, u_k, v_k, n = arrow.long()
                source_idx = u_s * N + v_s
                sink_idx = u_k * N + v_k
                
                if source_idx < N*N and sink_idx < N*N:
                    R[source_idx, sink_idx] += n.float()
            
            if R.sum() == 0:
                results.append(None)
                continue
                
            R_sum = R.sum()
            R_norm = R / R_sum
            sparsity = (R == 0).float().mean().item()
            num_arrows = (arrows[:, 0] != -1).sum().item()
            
            # Full matrix SVD on GPU
            s_full = torch.linalg.svdvals(R_norm).cpu()
            
            # Reduced matrix SVD (non-zero rows/cols only)
            nonzero_rows = (R.sum(dim=1) > 0)
            nonzero_cols = (R.sum(dim=0) > 0)
            
            s_reduced = None
            if nonzero_rows.sum() > 1 and nonzero_cols.sum() > 1:
                R_sub = R_norm[nonzero_rows][:, nonzero_cols]
                s_reduced = torch.linalg.svdvals(R_sub).cpu()
            
            results.append({
                'full_sv': s_full,
                'reduced_sv': s_reduced,
                'n_atoms': N,
                'n_arrows': num_arrows,
                'sparsity': sparsity,
                'nonzero_entries': (R > 0).sum().item()
            })
        except Exception as e:
            print(f"Error processing sample {b}: {e}")
            results.append(None)
    
    return results

def main():
    path = "data/flower_small/val.pt"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    print(f"Loading {path}...")
    data_list = torch.load(path, map_location='cpu', weights_only=False)
    
    # Handle dict of lists format
    if isinstance(data_list, dict):
        print("Detected dictionary of lists format.")
        samples = []
        keys = list(data_list.keys())
        print("Converting dict to list of samples...")
        for i in tqdm(range(len(data_list[keys[0]])), desc="Loading"):
            samples.append({k: data_list[k][i] for k in keys})
    else:
        samples = data_list

    num_samples = len(samples)
    print(f"Dataset contains {num_samples} samples.")
    
    # Estimate average molecule size
    sample_size = min(10, num_samples)
    avg_atoms = np.mean([torch.as_tensor(samples[i]['src_matrices']).shape[0] 
                         for i in range(sample_size)])
    print(f"Estimated average atoms: {avg_atoms:.1f}")
    print(f"Estimated matrix size: {int(avg_atoms**2)} x {int(avg_atoms**2)}")
    
    limit = min(2000, num_samples)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        batch_size = 4  # Process 4 samples at a time on GPU
    else:
        print("\n⚠ No GPU detected, using CPU (will be slower)")
        batch_size = 1
    
    print(f"\nProcessing {limit} samples on {device} in batches of {batch_size}...")
    
    # Prepare work items and batch them
    work_items = [(b, samples[b]) for b in range(limit)]
    batches = [work_items[i:i+batch_size] for i in range(0, len(work_items), batch_size)]
    
    # Process batches with progress bar
    results_raw = []
    for batch in tqdm(batches, desc="Computing SVDs", unit="batch"):
        batch_results = process_sample_batch_gpu(batch, device)
        results_raw.extend(batch_results)
    
    # Filter out None results
    results = [r for r in results_raw if r is not None]
    size_bins = defaultdict(list)
    
    # Bin by size
    for result in results:
        size_bin = (result['n_atoms'] // 5) * 5
        size_bins[size_bin].append(result)

    if not results:
        print("\nNo valid electron transfers found.")
        return
    
    print(f"\n{'='*60}")
    print(f"Successfully processed {len(results)} samples with valid reactions.")
    print(f"{'='*60}")
    print(" DATASET STATISTICS")
    print("="*60)
    
    avg_sparsity = np.mean([r['sparsity'] for r in results])
    avg_atoms = np.mean([r['n_atoms'] for r in results])
    avg_arrows = np.mean([r['n_arrows'] for r in results])
    avg_nonzero = np.mean([r['nonzero_entries'] for r in results])
    
    print(f"Average atoms per molecule: {avg_atoms:.1f}")
    print(f"Average arrows per reaction: {avg_arrows:.1f}")
    print(f"Average matrix sparsity: {avg_sparsity*100:.2f}%")
    print(f"Average non-zero entries: {avg_nonzero:.1f}")
    
    # Analyze FULL matrix singular values
    print("\n" + "="*60)
    print(" FULL MATRIX RANK ANALYSIS")
    print("="*60)
    print("Computing average singular values across samples...")
    
    max_len_full = max(len(r['full_sv']) for r in results)
    padded_full = [torch.cat([r['full_sv'], torch.zeros(max_len_full - len(r['full_sv']))]) 
                   for r in results]
    avg_s_full = torch.stack(padded_full).mean(dim=0)
    
    s_sq_full = avg_s_full ** 2
    total_var_full = s_sq_full.sum()
    explained_var_full = s_sq_full / total_var_full
    cum_var_full = torch.cumsum(explained_var_full, dim=0)
    
    print(f"Rank 1 explains: {explained_var_full[0]*100:.2f}%")
    print(f"Rank 2 explains: {explained_var_full[1]*100:.2f}%")
    print(f"Rank 5 explains: {cum_var_full[4]*100:.2f}%")
    print(f"Rank 10 explains: {cum_var_full[9]*100:.2f}%")
    
    rank_90 = (cum_var_full >= 0.90).nonzero()
    rank_95 = (cum_var_full >= 0.95).nonzero()
    if len(rank_90) > 0:
        print(f"Rank for 90% variance: {rank_90[0].item() + 1}")
    if len(rank_95) > 0:
        print(f"Rank for 95% variance: {rank_95[0].item() + 1}")
    
    # Analyze REDUCED matrix singular values
    print("\n" + "="*60)
    print(" REDUCED MATRIX RANK ANALYSIS (non-zero support only)")
    print("="*60)
    
    reduced_results = [r for r in results if r['reduced_sv'] is not None]
    print(f"Analyzing {len(reduced_results)} samples with valid reduced matrices...")
    
    if reduced_results:
        max_len_reduced = max(len(r['reduced_sv']) for r in reduced_results)
        padded_reduced = [torch.cat([r['reduced_sv'], torch.zeros(max_len_reduced - len(r['reduced_sv']))]) 
                         for r in reduced_results]
        avg_s_reduced = torch.stack(padded_reduced).mean(dim=0)
        
        s_sq_reduced = avg_s_reduced ** 2
        total_var_reduced = s_sq_reduced.sum()
        explained_var_reduced = s_sq_reduced / total_var_reduced
        cum_var_reduced = torch.cumsum(explained_var_reduced, dim=0)
        
        print(f"Rank 1 explains: {explained_var_reduced[0]*100:.2f}%")
        print(f"Rank 2 explains: {explained_var_reduced[1]*100:.2f}%")
        if len(cum_var_reduced) > 4:
            print(f"Rank 5 explains: {cum_var_reduced[4]*100:.2f}%")
        if len(cum_var_reduced) > 9:
            print(f"Rank 10 explains: {cum_var_reduced[9]*100:.2f}%")
        
        rank_90_red = (cum_var_reduced >= 0.90).nonzero()
        rank_95_red = (cum_var_reduced >= 0.95).nonzero()
        if len(rank_90_red) > 0:
            print(f"Rank for 90% variance: {rank_90_red[0].item() + 1}")
        if len(rank_95_red) > 0:
            print(f"Rank for 95% variance: {rank_95_red[0].item() + 1}")
    
    # Create comprehensive visualization
    print("\nGenerating visualizations...")
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Full matrix singular values
    plt.subplot(2, 3, 1)
    plt.semilogy(range(1, min(21, len(avg_s_full)+1)), avg_s_full[:20].numpy(), 'o-')
    plt.xlabel('Rank')
    plt.ylabel('Singular Value')
    plt.title('Full Matrix: Singular Values (log scale)')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Full matrix cumulative variance
    plt.subplot(2, 3, 2)
    plt.plot(range(1, min(21, len(cum_var_full)+1)), cum_var_full[:20].numpy(), 's-', color='orange')
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
    plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95%')
    plt.xlabel('Rank')
    plt.ylabel('Cumulative Variance Explained')
    plt.title('Full Matrix: Cumulative Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Reduced matrix singular values
    if reduced_results:
        plt.subplot(2, 3, 3)
        plt.semilogy(range(1, min(21, len(avg_s_reduced)+1)), avg_s_reduced[:20].numpy(), 'o-', color='green')
        plt.xlabel('Rank')
        plt.ylabel('Singular Value')
        plt.title('Reduced Matrix: Singular Values (log scale)')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Reduced matrix cumulative variance
    if reduced_results:
        plt.subplot(2, 3, 4)
        plt.plot(range(1, min(21, len(cum_var_reduced)+1)), cum_var_reduced[:20].numpy(), 's-', color='purple')
        plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='90%')
        plt.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95%')
        plt.xlabel('Rank')
        plt.ylabel('Cumulative Variance Explained')
        plt.title('Reduced Matrix: Cumulative Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Sparsity distribution
    plt.subplot(2, 3, 5)
    sparsities = [r['sparsity'] for r in results]
    plt.hist(sparsities, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Sparsity')
    plt.ylabel('Count')
    plt.title('Matrix Sparsity Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Size vs rank quality
    plt.subplot(2, 3, 6)
    sizes = [r['n_atoms'] for r in results]
    rank1_vars = [((r['full_sv'][0]**2) / (r['full_sv']**2).sum()).item() for r in results]
    plt.scatter(sizes, rank1_vars, alpha=0.5)
    plt.xlabel('Number of Atoms')
    plt.ylabel('Rank-1 Variance Explained')
    plt.title('Molecule Size vs Low-Rank Quality')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("comprehensive_rank_analysis.png", dpi=150)
    print("\n" + "="*60)
    print("✓ Plot saved to 'comprehensive_rank_analysis.png'")
    print("="*60)
    
    # Analysis by molecule size
    if len(size_bins) > 1:
        print("\n" + "="*60)
        print(" ANALYSIS BY MOLECULE SIZE")
        print("="*60)
        for size_bin in sorted(size_bins.keys()):
            bin_results = size_bins[size_bin]
            if len(bin_results) < 5:
                continue
            
            bin_sv = [r['full_sv'] for r in bin_results]
            max_len = max(len(sv) for sv in bin_sv)
            padded = [torch.cat([sv, torch.zeros(max_len - len(sv))]) for sv in bin_sv]
            avg_sv = torch.stack(padded).mean(dim=0)
            
            var_explained = ((avg_sv[0]**2) / (avg_sv**2).sum()).item()
            
            print(f"Size {size_bin}-{size_bin+4} atoms (n={len(bin_results)}): Rank-1 explains {var_explained*100:.2f}%")

if __name__ == "__main__":
    main()
