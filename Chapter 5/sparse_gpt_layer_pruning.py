import torch
import torch.nn as nn
import numpy as np

def sparse_gpt_layer_pruning(layer, sparsity_level=0.5, sample_inputs=None):
    if not isinstance(layer, nn.Linear):
        print("This function is designed for nn.Linear layers.")
        return

    W = layer.weight.data.clone().to(torch.float64)
    rows, cols = W.shape

    if sample_inputs is None:
        sample_inputs = torch.randn(128, cols).to(torch.float64)
    else:
        sample_inputs = sample_inputs.to(torch.float64)

    H = sample_inputs.t() @ sample_inputs

    W_pruned = W.clone()
    
    errors = torch.zeros(cols, dtype=torch.float64)
    pruned_mask = torch.ones_like(W_pruned)

    for j in range(cols):
        w_col = W_pruned[:, j]
        h_inv_diag = 1.0 / (H[j, j] + 1e-9)

        num_to_prune = int(sparsity_level * rows)
        if num_to_prune > 0:
            threshold = torch.kthvalue(w_col.abs(), num_to_prune).values
            prune_indices = w_col.abs() <= threshold
            
            error = (w_col ** 2) * h_inv_diag
            errors[j] = error.sum()

            w_col[prune_indices] = 0
            pruned_mask[:, j][prune_indices] = 0

    for i in range(rows):
        unpruned_indices = torch.where(pruned_mask[i, :] == 1)[0]
        
        if len(unpruned_indices) > 0 and len(unpruned_indices) < cols:
            W_orig_row = W[i, :]
            H_sub = H[np.ix_(unpruned_indices, unpruned_indices)]
            W_H_sub = W_orig_row @ H[:, unpruned_indices]
            
            try:
                updated_weights_sub = torch.linalg.solve(H_sub, W_H_sub.t()).t()
                W_pruned[i, unpruned_indices] = updated_weights_sub
            except torch.linalg.LinAlgError:
                pass

    layer.weight.data = W_pruned.to(layer.weight.dtype)
    print(f"Layer pruned to ~{sparsity_level*100}% sparsity.")
    print(f"Original non-zero count: {torch.count_nonzero(W)}")
    print(f"Pruned non-zero count: {torch.count_nonzero(layer.weight.data)}")

d_model = 512
sample_layer = nn.Linear(d_model, d_model)
original_norm = torch.norm(sample_layer.weight.data)

sample_activations = torch.randn(128, d_model) 

sparse_gpt_layer_pruning(sample_layer, sparsity_level=0.6, sample_inputs=sample_activations)

pruned_norm = torch.norm(sample_layer.weight.data)
print(f"Original weight norm: {original_norm:.4f}")
print(f"Pruned weight norm: {pruned_norm:.4f}")