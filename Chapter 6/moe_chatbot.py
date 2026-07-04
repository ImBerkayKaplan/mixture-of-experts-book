import torch
import torch.nn as nn
import random

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.layer = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.layer(x)
        routing_weights = self.softmax(logits)
        return routing_weights

class MoEChatbot(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_experts):
        super(MoEChatbot, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        self.gating_network = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        pooled_x = x.mean(dim=1)
        routing_weights = self.gating_network(pooled_x)
        
        expert_indices = torch.argmax(routing_weights, dim=1)
        
        batch_size = x.size(0)
        seq_len = x.size(1)
        output_dim = self.experts[0].network[-1].out_features
        output = torch.zeros(batch_size, seq_len, output_dim, device=x.device, dtype=x.dtype)

        for i in range(batch_size):
            chosen_expert_index = expert_indices[i].item()
            chosen_expert = self.experts[chosen_expert_index]
            output[i] = chosen_expert(x[i].unsqueeze(0))
            
        return output

class SimpleTokenizer:
    def __init__(self):
        self.vocab = {'<pad>': 0, '<unk>': 1, 'hello': 2, 'hi': 3, 'how': 4, 'are': 5, 'you': 6,
                      'what': 7, 'is': 8, 'the': 9, 'weather': 10, 'like': 11, 'today': 12, '?': 13,
                      '.': 14, 'i': 15, 'am': 16, 'fine': 17, 'thank': 18, 'it': 19, 'sunny': 20}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def encode(self, text, max_len=10):
        tokens = text.lower().split()
        encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        padded = encoded + [self.vocab['<pad>']] * (max_len - len(encoded))
        return torch.tensor(padded[:max_len], dtype=torch.long)

    def decode(self, tensor):
        tokens = [self.inverse_vocab.get(idx.item(), '<unk>') for idx in tensor]
        return ' '.join(token for token in tokens if token != '<pad>')

def get_mock_data_loader(tokenizer, batch_size=4):
    data = [
        ("hello", "hi there"),
        ("how are you ?", "i am a bot i am fine"),
        ("what is the weather like ?", "it is sunny today ."),
        ("hi", "hello how can i help ?"),
    ]
    
    inputs, targets = zip(*data)
    input_tensors = torch.stack([tokenizer.encode(t) for t in inputs])
    target_tensors = torch.stack([tokenizer.encode(t) for t in targets])
    
    dataset = torch.utils.data.TensorDataset(input_tensors, target_tensors)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def main():
    # Hyperparameters
    input_dim = 32
    hidden_dim = 64
    output_dim = 32
    num_experts = 4
    max_len = 10
    
    # Initialization
    tokenizer = SimpleTokenizer()
    embedding = nn.Embedding(tokenizer.vocab_size, input_dim)
    model = MoEChatbot(input_dim, hidden_dim, output_dim, num_experts)
    data_loader = get_mock_data_loader(tokenizer)
    
    # Training Setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Simplified Training Loop
    print("Starting simplified training...")
    for epoch in range(50):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            
            input_embeds = embedding(inputs)
            target_embeds = embedding(targets)
            
            output_embeds = model(input_embeds)
            
            loss = criterion(output_embeds, target_embeds)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")

    # Chat Simulation
    print("\nTraining complete. Starting chat simulation.")
    model.eval()
    with torch.no_grad():
        test_input = "how are you ?"
        print(f"User: {test_input}")
        
        encoded_input = tokenizer.encode(test_input).unsqueeze(0)
        input_embeds = embedding(encoded_input)
        
        # Manually trace the routing using pooled embeddings, matching model.forward().
        pooled_input = input_embeds.mean(dim=1)
        routing_weights = model.gating_network(pooled_input)
        chosen_expert = torch.argmax(routing_weights, dim=1).item()
        print(f"Gating Network routed to Expert: {chosen_expert}")

        output_embeds = model(input_embeds)
        
        # For this simulation, we find the closest vocab vector as a pseudo-decode
        flat_vocab_embeds = embedding.weight.unsqueeze(0)
        distances = torch.cdist(output_embeds, flat_vocab_embeds)
        output_indices = torch.argmin(distances, dim=2).squeeze(0)

        response = tokenizer.decode(output_indices)
        print(f"Chatbot: {response}")

if __name__ == '__main__':
    main()