import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# --- Healthcare Application Components ---

def generate_patient_data(num_samples=1000, num_features=10):
    """Simulates patient data for a healthcare MoE."""
    features = np.random.rand(num_samples, num_features) * 100
    contexts = np.random.randint(0, 3, size=num_samples)

    labels = np.zeros(num_samples)
    for i in range(num_samples):
        if contexts[i] == 0:  # Cardiology context
            labels[i] = 0.2 * features[i, 0] + 0.8 * features[i, 2]
        elif contexts[i] == 1:  # Oncology context
            labels[i] = 0.5 * features[i, 1] - 0.3 * features[i, 4]
        else:  # Diabetes context
            labels[i] = 0.9 * features[i, 3] + 0.1 * features[i, 5]

    # Normalize labels to be in a [0, 1] range for stability
    max_val = np.max(np.abs(labels))
    labels = (labels / max_val + 1) / 2

    features_tensor = torch.tensor(features, dtype=torch.float32)
    contexts_tensor = torch.tensor(contexts, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(features_tensor, contexts_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    return loader

class HealthcareExpert(nn.Module):
    """Expert network for the healthcare model."""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(HealthcareExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

class HealthcareGatingNetwork(nn.Module):
    """Gating network for the healthcare model."""
    def __init__(self, input_dim, num_experts):
        super(HealthcareGatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.layer(x)
        routing_weights = self.softmax(logits)
        return routing_weights

class MoEDecisionSystem(nn.Module):
    """Top-level MoE model for healthcare decisions."""
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(MoEDecisionSystem, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            HealthcareExpert(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.gating = HealthcareGatingNetwork(input_dim, num_experts)

    def forward(self, x):
        routing_weights = self.gating(x)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        routing_weights_unsqueezed = routing_weights.unsqueeze(1)
        weighted_output = torch.bmm(routing_weights_unsqueezed, expert_outputs.transpose(1, 2)).squeeze(1)
        return weighted_output, routing_weights

def train_and_evaluate_healthcare():
    """Main function to run the healthcare MoE simulation."""
    print("--- Starting Healthcare MoE Simulation ---")
    input_dim = 10
    hidden_dim = 32
    num_experts = 3
    num_epochs = 20

    data_loader = generate_patient_data()
    model = MoEDecisionSystem(input_dim, hidden_dim, num_experts)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for features, contexts, labels in data_loader:
            optimizer.zero_grad()
            output, _ = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    print("Training finished.\n")

    model.eval()
    test_loader = generate_patient_data(num_samples=100)
    features, contexts, labels = next(iter(test_loader))

    with torch.no_grad():
        output, routing_weights = model(features)
        print("Example Patient Predictions (Contexts: 0=Cardiology, 1=Oncology, 2=Diabetes)")
        for i in range(5):
            print(f"\nPatient Context: {contexts[i].item()}")
            print(f"Expert Weights: {routing_weights[i].numpy().round(2)}")
            print(f"Predicted Risk: {output[i].item():.4f}, Actual Risk: {labels[i].item():.4f}")
    print("\n--- Healthcare Simulation Finished ---\n")


# --- Financial Application Components ---

def generate_transaction_data(num_samples=5000, num_features=8):
    """Simulates financial transaction data for a fraud detection MoE."""
    features = np.random.rand(num_samples, num_features)
    contexts = np.random.randint(0, 3, size=num_samples)
    labels = np.zeros(num_samples)

    for i in range(num_samples):
        is_fraud = 0
        if contexts[i] == 0:  # Online transaction
            if features[i, 0] > 0.9 and features[i, 2] < 0.1:
                is_fraud = 1
        elif contexts[i] == 1:  # In-store transaction
            if features[i, 1] > 0.95 and features[i, 4] > 0.95:
                is_fraud = 1
        else:  # Wire Transfer
            if features[i, 3] > 0.98:
                is_fraud = 1

        if np.random.rand() < 0.05:  # Add some noise to the labels
            is_fraud = 1 - is_fraud
        labels[i] = is_fraud

    features_tensor = torch.tensor(features, dtype=torch.float32)
    contexts_tensor = torch.tensor(contexts, dtype=torch.long)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(features_tensor, contexts_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

class FinancialExpert(nn.Module):
    """Expert network for the fraud detection model."""
    def __init__(self, input_dim, hidden_dim):
        super(FinancialExpert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

class FinancialGatingNetwork(nn.Module):
    """Gating network for the fraud detection model."""
    def __init__(self, input_dim, num_experts):
        super(FinancialGatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.softmax(self.layer(x))

class MoEFraudDetector(nn.Module):
    """Top-level MoE model for fraud detection."""
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(MoEFraudDetector, self).__init__()
        self.experts = nn.ModuleList([
            FinancialExpert(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.gating = FinancialGatingNetwork(input_dim, num_experts)

    def forward(self, x):
        routing_weights = self.gating(x)
        expert_outputs = torch.cat([expert(x) for expert in self.experts], dim=1)
        final_output = torch.sum(routing_weights * expert_outputs, dim=1, keepdim=True)
        return final_output, routing_weights

def train_and_detect_fraud():
    """Main function to run the fraud detection MoE simulation."""
    print("--- Starting Financial Fraud Detection MoE Simulation ---")
    input_dim = 8
    hidden_dim = 24
    num_experts = 3
    num_epochs = 15

    data_loader = generate_transaction_data()
    model = MoEFraudDetector(input_dim, hidden_dim, num_experts)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    for epoch in range(num_epochs):
        for features, contexts, labels in data_loader:
            optimizer.zero_grad()
            output, _ = model(features)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

    print("Fraud detection model training finished.\n")

    model.eval()
    test_loader = generate_transaction_data(num_samples=200)
    features, contexts, labels = next(iter(test_loader))

    with torch.no_grad():
        output, routing_weights = model(features)
        print("Example Transaction Predictions (Contexts: 0=Online, 1=In-Store, 2=Wire)")
        for i in range(5):
            pred_label = "Fraud" if output[i].item() > 0.5 else "Not Fraud"
            actual_label = "Fraud" if labels[i].item() == 1 else "Not Fraud"
            print(f"\nTransaction Context: {contexts[i].item()}")
            print(f"Expert Weights: {routing_weights[i].numpy().round(2)}")
            print(f"Prediction: {pred_label}, Actual: {actual_label}")
    print("\n--- Financial Simulation Finished ---")


if __name__ == '__main__':
    train_and_evaluate_healthcare()
    train_and_detect_fraud()