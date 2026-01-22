import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    path = "data/flower_small/val.pt"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    print(f"Loading {path}...")
    # Loading the full cache. Based on your error, this is a list of samples.
    data_list = torch.load(path, map_location='cpu', weights_only=False)
    
    # If the file is a dict of lists instead of a list of dicts, handle that too
    if isinstance(data_list, dict):
        print("Detected dictionary of lists format.")
        samples = []
        keys = list(data_list.keys())
        for i in range(len(data_list[keys[0]])):
            samples.append({k: data_list[k][i] for k in keys})
    else:
        samples = data_list

    num_samples = len(samples)
    print(f"Processing {num_samples} samples...")
        
    all_singular_values = []
    limit = min(1000, num_samples)
    
    for b in range(limit):
        sample = samples[b]
        
        # Convert whatever format (list/array) to torch tensor
        src_matrix = torch.as_tensor(sample['src_matrices'])
        arrows = torch.as_tensor(sample['src_arrows'])
        
        # N is the number of atoms
        N = src_matrix.shape[0]
        
        # Transition matrix R (N^2 x N^2)
        R = torch.zeros((N * N, N * N))
        
        # Handle arrows (u_s, v_s, u_k, v_k, count)
        # If it's a 2D tensor, iterate rows. If 1D, it's just one arrow.
        if arrows.ndim == 1:
            arrows = arrows.unsqueeze(0)

        for arrow in arrows:
            if arrow[0] == -1: break # Padding check
            
            u_s, v_s, u_k, v_k, n = arrow.long()
            
            # Map coordinates to flattened N^2 space
            source_idx = u_s * N + v_s
            sink_idx = u_k * N + v_k
            
            # Bound check to avoid index errors on small/large N
            if source_idx < N*N and sink_idx < N*N:
                R[source_idx, sink_idx] += n.float()
            
        if R.sum() > 0:
            s = torch.linalg.svdvals(R)
            all_singular_values.append(s)

    if not all_singular_values:
        print("No valid electron transfers found in the sample limit.")
        return
        
    # Standardize singular value lengths for averaging
    max_s_len = max(len(s) for s in all_singular_values)
    padded_s = [torch.cat([s, torch.zeros(max_s_len - len(s))]) for s in all_singular_values]
    
    avg_s = torch.stack(padded_s).mean(dim=0)
    
    # Rank Analysis
    s_sq = avg_s ** 2
    total_var = s_sq.sum()
    explained_var = s_sq / total_var
    cum_var = torch.cumsum(explained_var, dim=0)

    print("\n" + "="*30)
    print(" SVD RANK ANALYSIS RESULTS")
    print("="*30)
    print(f"Rank 1 explains: {explained_var[0]*100:.2f}% of variance")
    print(f"Rank 2 explains: {explained_var[1]*100:.2f}% of variance")
    
    # Save the scree plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, 11), avg_s[:10].numpy(), 'o-')
    plt.title("Singular Values")
    plt.subplot(1, 2, 2)
    plt.plot(range(1, 11), cum_var[:10].numpy(), 's-', color='orange')
    plt.title("Cumulative Variance")
    plt.savefig("rank_analysis_flower.png")
    print("\nPlot saved to 'rank_analysis_flower.png'")

if __name__ == "__main__":
    main()
