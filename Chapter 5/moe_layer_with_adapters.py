import torch.nn.functional as F
import torch.nn as nn
import torch

class Expert(nn.Module): 
    def __init__(self, d_model, d_ff): 
        super().__init__() 
        self.fc1 = nn.Linear(d_model, d_ff) 
        self.activation = nn.GELU() 
        self.fc2 = nn.Linear(d_ff, d_model) 

    def forward(self, x): 
        return self.fc2(self.activation(self.fc1(x))) 

class Adapter(nn.Module):
    def __init__(self, d_model, bottleneck_dim):
        super().__init__()
        self.down_project = nn.Linear(d_model, bottleneck_dim)
        self.up_project = nn.Linear(bottleneck_dim, d_model)
        self.activation = nn.GELU()
        
        nn.init.zeros_(self.up_project.weight)
        nn.init.zeros_(self.up_project.bias)
 
    def forward(self, x):
        adapter_output = self.up_project(self.activation(self.down_project(x)))
        return x + adapter_output

class ExpertWithAdapter(nn.Module): 
    def __init__(self, d_model, d_ff, bottleneck_dim): 
        super().__init__() 
        self.base_expert = Expert(d_model, d_ff) 
        self.adapter = Adapter(d_model, bottleneck_dim) 

    def forward(self, x): 
        expert_output = self.base_expert(x) 
        adapter_output = self.adapter(expert_output) 
        return expert_output + adapter_output 

class MoELayerWithAdapters(nn.Module):
    def __init__(self, d_model, num_experts, d_ff, bottleneck_dim, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
 
        self.gate = nn.Linear(d_model, num_experts)
 
        self.experts = nn.ModuleList([
            ExpertWithAdapter(d_model, d_ff, bottleneck_dim) for _ in range(num_experts)
        ])
 
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.d_model)
        
        gate_logits = self.gate(x_flat)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
 
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.num_experts):
            token_indices, _ = torch.where(top_k_indices == i)
            
            if token_indices.numel() > 0:
                expert_input = x_flat[token_indices]
                expert_output = self.experts[i](expert_input)
                gating_weights = top_k_probs[token_indices, (top_k_indices[token_indices] == i).nonzero(as_tuple=True)[1]]
                final_output.index_add_(0, token_indices, expert_output * gating_weights.unsqueeze(-1))
 
        return final_output.view(batch_size, seq_len, self.d_model)
    
# Configuration
D_MODEL = 512
D_FF = 2048
NUM_EXPERTS = 8
BOTTLENECK_DIM = 64
TOP_K = 2
 
moe_adapter_model = MoELayerWithAdapters(
    d_model=D_MODEL,
    num_experts=NUM_EXPERTS,
    d_ff=D_FF,
    bottleneck_dim=BOTTLENECK_DIM,
    top_k=TOP_K
)
 
# Freeze all parameters in the model initially
for param in moe_adapter_model.parameters():
    param.requires_grad = False
 
# Unfreeze ONLY the adapter parameters
for expert in moe_adapter_model.experts:
    for param in expert.adapter.parameters():
        param.requires_grad = True
 
# Verify which parameters are trainable
total_params = 0
trainable_params = 0
for name, param in moe_adapter_model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
 
print(f"Total model parameters: {total_params:,}")
print(f"Trainable adapter parameters: {trainable_params:,}")
print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.4f}%")
 
dummy_input = torch.randn(4, 10, D_MODEL)
output = moe_adapter_model(dummy_input)