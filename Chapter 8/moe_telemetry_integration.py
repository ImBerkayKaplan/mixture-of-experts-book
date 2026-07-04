# ------------------------- Not included in the book -------------------------

import numpy as np
import torch
from torch import nn


class SimpleMoEBlock(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x):
        batch_size = x.shape[0]
        expert_indices = torch.arange(batch_size) % self.num_experts
        return x, expert_indices.to(torch.long)


class SimpleLayer(nn.Module):
    def __init__(self, num_experts=4):
        super().__init__()
        self.moe_block = SimpleMoEBlock(num_experts=num_experts)

    def forward(self, x):
        output = self.moe_block(x)
        return output[0]


class SimpleTransformer(nn.Module):
    def __init__(self, num_layers=5, num_experts=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [SimpleLayer(num_experts=num_experts) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MinimalMoEModel(nn.Module):
    def __init__(self, num_layers=5, num_experts=4):
        super().__init__()
        self.transformer = SimpleTransformer(
            num_layers=num_layers, num_experts=num_experts
        )

    def forward(self, x):
        return self.transformer(x)

model = MinimalMoEModel()

# ------------------------- Not included in the book -------------------------
import mlflow

def make_routing_logger_hook(layer_name):
    def hook(module, input, output):
        # output[1] contains the selected expert indices from the router
        expert_indices = output[1].detach().cpu().numpy()

        # Calculate utilization frequencies
        counts = np.bincount(expert_indices.flatten(), minlength=module.num_experts)
        utilization_dict = {
            f"{layer_name}_expert_{i}_count": int(c) for i, c in enumerate(counts)
        }

        # Push telemetry data directly to cloud-managed tracking engines
        mlflow.log_metrics(utilization_dict)

        print(f"[telemetry] {layer_name} expert counts: {counts.tolist()}")

    return hook

model.transformer.layers[4].moe_block.register_forward_hook(
    make_routing_logger_hook(layer_name="layer_4_moe")
)

mlflow.set_experiment("minimal_moe_telemetry")
with mlflow.start_run():
    torch.manual_seed(0)
    sample_input = torch.randn(8, 16)
    print(f"Input shape: {tuple(sample_input.shape)}")
    model(sample_input)