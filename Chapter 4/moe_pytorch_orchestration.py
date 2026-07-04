import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    def __init__(self, d_model: int, num_experts: int, top_k: int, d_ffn: int, dropout: float = 0.1, aux_loss_alpha: float = 0.01):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha

        self.gate = nn.Linear(d_model, num_experts, bias=False)

        self.experts = nn.ModuleList([Expert(d_model, d_ffn, dropout) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, d_model = x.shape
        
        x_reshaped = x.view(-1, d_model)
        num_tokens = x_reshaped.shape[0]
        gate_logits = self.gate(x_reshaped)

        routing_weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float)
        
        mask = torch.zeros_like(gate_logits, dtype=torch.float).scatter_(-1, selected_experts, 1)
        
        tokens_per_expert = mask.sum(dim=0)
        fraction_tokens_per_expert = tokens_per_expert / num_tokens
        
        probs_per_expert = F.softmax(gate_logits, dim=-1, dtype=torch.float).mean(dim=0)
        aux_loss = self.aux_loss_alpha * self.num_experts * torch.dot(fraction_tokens_per_expert, probs_per_expert)

        final_output = torch.zeros_like(x_reshaped)
        flat_expert_indices = selected_experts.flatten()
        flat_token_indices = torch.arange(num_tokens, device=x.device).repeat_interleave(self.top_k)
        
        expert_inputs = x_reshaped[flat_token_indices]
        expert_outputs = torch.zeros_like(expert_inputs)

        for i in range(self.num_experts):
            idx = (flat_expert_indices == i)
            if idx.any():
                expert_outputs[idx] = self.experts[i](expert_inputs[idx])

        flat_routing_weights = routing_weights.flatten().unsqueeze(1)
        weighted_outputs = expert_outputs * flat_routing_weights
        
        final_output.index_add_(0, flat_token_indices, weighted_outputs)
        final_output = final_output.view(batch_size, seq_len, d_model)
        return final_output, aux_loss