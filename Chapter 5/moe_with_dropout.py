import torch
import torch.nn as nn
import torch.nn.functional as F
 
class MoELayerWithExpertDropout(nn.Module):
    def __init__(self, d_model, num_experts, d_ff, top_k=2, dropout_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.dropout_prob = dropout_prob
        self.gate = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])
 
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        gate_logits = self.gate(x_flat)
 
        dropped_mask = None
        if self.training and self.dropout_prob > 0:
            num_dropped = int(self.num_experts * self.dropout_prob)
            if num_dropped > 0:
                dropped_indices = torch.randperm(self.num_experts, device=gate_logits.device)[:num_dropped]
                dropped_mask = torch.zeros(self.num_experts, dtype=torch.bool, device=gate_logits.device)
                dropped_mask[dropped_indices] = True
                gate_logits = gate_logits.masked_fill(dropped_mask.unsqueeze(0), float('-inf'))
        gate_probs = F.softmax(gate_logits, dim=-1) 
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1) 
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True) 
        final_output = torch.zeros_like(x_flat) 
        for i in range(self.num_experts): 
            if self.training and self.dropout_prob > 0 and dropped_mask is not None and dropped_mask[i]: 
                continue 
            token_indices, = torch.where(torch.any(top_k_indices == i, dim=-1)) 
            if token_indices.numel() > 0: 
                expert_input = x_flat[token_indices] 
                expert_output = self.experts[i](expert_input)
                expert_mask = top_k_indices[token_indices] == i 
                gating_weights = top_k_probs[token_indices][expert_mask] 
                final_output.index_add_(0, token_indices, expert_output * gating_weights.unsqueeze(-1))
        return final_output.view(batch_size, seq_len, self.d_model) 
 
    def forward_with_temperature(self, gate_logits, temperature=1.5):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        scaled_logits = gate_logits / temperature
        gate_probs = F.softmax(scaled_logits, dim=-1)
        return gate_probs

# 1) Create the model
model = MoELayerWithExpertDropout(
    d_model=16,
    num_experts=4,
    d_ff=64,
    top_k=2,
    dropout_prob=0.25,
)

# 2) Create a batch of dummy input
batch_size = 2
seq_len = 5
x = torch.randn(batch_size, seq_len, 16)

# 3) Forward pass in training mode
model.train()
output_train = model(x)
print("train output shape:", output_train.shape)

# 4) Forward pass in eval mode (no expert dropout)
model.eval()
output_eval = model(x)
print("eval output shape:", output_eval.shape)