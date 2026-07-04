import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def generate_fact_check_data(num_samples=10000, num_features=12):
    features = np.random.randn(num_samples, num_features)
    contexts = np.random.randint(0, 3, size=num_samples)
    labels = np.zeros(num_samples)

    for i in range(num_samples):
        is_true = 0
        if contexts[i] == 0:
            if features[i, 1] > 0.8 and features[i, 4] < -0.5:
                is_true = 1
        elif contexts[i] == 1:
            if features[i, 2] > 0.9 and features[i, 7] > 0.9:
                is_true = 1
        else:
            if features[i, 5] < -0.9 and features[i, 10] > 0.7:
                is_true = 1

        if np.random.rand() < 0.15:
            is_true = 1 - is_true

        labels[i] = is_true

    features_tensor = torch.tensor(features, dtype=torch.float32)
    contexts_tensor = torch.tensor(contexts, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(features_tensor, contexts_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

class FactCheckingExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FactCheckingExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class FactCheckingGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(FactCheckingGatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.layer(x)
        routing_weights = self.softmax(logits)
        return routing_weights

class MoEFactChecker(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(MoEFactChecker, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            FactCheckingExpert(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.gating = FactCheckingGatingNetwork(input_dim, num_experts)

    def forward(self, x):
        routing_weights = self.gating(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        routing_weights_expanded = routing_weights.unsqueeze(1)
        weighted_output = torch.bmm(routing_weights_expanded, expert_outputs).squeeze(1)
        return weighted_output, routing_weights

def train_and_evaluate_fact_checker():
    input_dim = 12
    hidden_dim = 48
    num_experts = 3
    num_epochs = 25

    data_loader = generate_fact_check_data()
    model = MoEFactChecker(input_dim, hidden_dim, num_experts)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Starting fact-checker model training...")
    for epoch in range(num_epochs):
        for features, _, labels in data_loader:
            optimizer.zero_grad()
            output, _ = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
    
    print("Training finished.\n")

    model.eval()
    test_loader = generate_fact_check_data(num_samples=100)
    features, contexts, labels = next(iter(test_loader))

    with torch.no_grad():
        output, routing_weights = model(features)
        for i in range(10):
            context_map = {0: 'Politics', 1: 'Science', 2: 'History'}
            pred_label = "True" if output[i].item() > 0.5 else "False"
            actual_label = "True" if labels[i].item() == 1 else "False"

            print(f"Claim Context: {context_map[contexts[i].item()]}")
            print(f"Expert Weights (P, S, H): {routing_weights[i].numpy().round(2)}")
            print(f"Prediction: {pred_label}, Actual: {actual_label}\n")

if __name__ == '__main__':
    train_and_evaluate_fact_checker()