import torch
import torch.nn as nn
import torch.nn.functional as F
from sparse_gpt_layer_pruning import sparse_gpt_layer_pruning

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, d_ff, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            token_indices, = torch.where(top_k_indices == i)
            
            if token_indices.numel() > 0:
                expert_input = x_flat[token_indices]
                expert_output = self.experts[i](expert_input)
                
                gating_weights = top_k_probs[token_indices, (top_k_indices[token_indices] == i).nonzero(as_tuple=True)[1]]
                
                final_output.index_add_(0, token_indices, expert_output * gating_weights.unsqueeze(-1))
                
        return final_output.view(batch_size, seq_len, self.d_model)

def prune_moe_layer_with_sparsegpt(moe_layer, expert_sparsity_map, calibration_data):
    print("--- Starting Expert-wise Pruning ---")
    
    for expert_idx, expert in enumerate(moe_layer.experts):
        if expert_idx in expert_sparsity_map:
            sparsity = expert_sparsity_map[expert_idx]
            print(f"\nPruning Expert {expert_idx} to {sparsity*100}% sparsity...")
            print("Pruning fc1 layer...")
            sparse_gpt_layer_pruning(expert.fc1, sparsity_level=sparsity, sample_inputs=calibration_data)
            
            with torch.no_grad():
                fc1_output = expert.activation(expert.fc1(calibration_data))
            
            print("Pruning fc2 layer...")
            sparse_gpt_layer_pruning(expert.fc2, sparsity_level=sparsity, sample_inputs=fc1_output)
            
            print(f"Expert {expert_idx} pruning complete.")
        else:
            print(f"\nSkipping Expert {expert_idx} (not in sparsity map).")

D_MODEL = 512
D_FF = 2048
NUM_EXPERTS = 8
TOP_K = 2

moe_model = MoELayer(d_model=D_MODEL, num_experts=NUM_EXPERTS, d_ff=D_FF, top_k=TOP_K)

sparsity_config = {
    0: 0.8,
    2: 0.5,
    5: 0.9,
    7: 0.4 
}

calibration_dataset = torch.randn(256, D_MODEL)

prune_moe_layer_with_sparsegpt(moe_model, sparsity_config, calibration_dataset)

total_params = 0
pruned_params = 0
for expert_idx, expert in enumerate(moe_model.experts):
    for param in expert.parameters():
        total_params += param.numel()
        pruned_params += torch.count_nonzero(param)

print("\n--- Pruning Summary ---")
print(f"Total parameters in experts (pre-pruning): {total_params}")
print(f"Parameters remaining in experts (post-pruning): {pruned_params}")
print(f"Overall expert sparsity: {100 * (1 - pruned_params / total_params):.2f}%")
