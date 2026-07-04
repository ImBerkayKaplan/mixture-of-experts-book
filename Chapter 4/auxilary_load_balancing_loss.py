import torch
import torch.nn.functional as F

def compute_auxiliary_loss(gating_outputs, num_experts, alpha=0.01):
    if gating_outputs is None or gating_outputs.numel() == 0:
        return torch.tensor(0.0, device=gating_outputs.device)
    gating_probs = F.softmax(gating_outputs, dim=-1)
    expert_indices = torch.argmax(gating_probs, dim=-1)
    token_to_expert_one_hot = F.one_hot(expert_indices, num_classes=num_experts).float()
    f_i = token_to_expert_one_hot.mean(dim=0)
    p_i = gating_probs.mean(dim=0)
    load_balancing_loss = (f_i * p_i).sum()
    aux_loss = alpha * num_experts * load_balancing_loss
    return aux_loss

num_tokens = 1024
num_experts = 8

gating_logits = torch.randn(num_tokens, num_experts)
gating_logits[:, 0] += 3.0
gating_logits[:, 1] += 2.5

aux_loss_value = compute_auxiliary_loss(gating_logits, num_experts)
print(f"Calculated Auxiliary Loss: {aux_loss_value.item():.6f}")

balanced_gating_logits = torch.randn(num_tokens, num_experts)
balanced_aux_loss = compute_auxiliary_loss(balanced_gating_logits, num_experts)
print(f"Calculated Auxiliary Loss (Balanced Gate): {balanced_aux_loss.item():.6f}")