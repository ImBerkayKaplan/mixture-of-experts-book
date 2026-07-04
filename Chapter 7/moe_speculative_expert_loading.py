import torch 
import time 
import random 

# ------------------------- Not included in the book -------------------------

class GatingNetwork(torch.nn.Module): 
    def forward(self, x): 
        time.sleep(0.05)  
        return random.randint(0, 7) 
class Expert(torch.nn.Module): 
    def __init__(self, id): 
        super().__init__() 
        self.id = id 
        self.layer = torch.nn.Linear(512, 512)  
    def forward(self, x): 
        return self.layer(x) 

# ------------------------- Not included in the book -------------------------

class SpeculativeMoEServer: 
    def __init__(self, num_experts=8): 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.gating_network = GatingNetwork().to(self.device) 
        self.cpu_experts = {i: Expert(i) for i in range(num_experts)} 
        self.gpu_expert_cache = {} 
        self.expert_access_counts = {i: 0 for i in range(num_experts)} 
        self.speculation_k = 2 # Speculatively load the top 2 experts 

    def _predict_experts(self): 
        sorted_experts = sorted(self.expert_access_counts.items(), key=lambda item: item[1], reverse=True) 
        return [expert_id for expert_id, count in sorted_experts[:self.speculation_k]] 

    def _load_expert_to_gpu(self, expert_id): 
        if expert_id not in self.gpu_expert_cache: 
            print(f"Loading expert {expert_id} to GPU...") 
            self.gpu_expert_cache[expert_id] = self.cpu_experts[expert_id].to(self.device) 

    def handle_request(self, input_data): 
        input_tensor = torch.randn(1, 512).to(self.device) # Mock input 
        predicted_ids = self._predict_experts() 
        print(f"Speculatively pre-warming experts: {predicted_ids}") 
        for expert_id in predicted_ids: 
            self._load_expert_to_gpu(expert_id) 
        print("Running gating network...") 
        chosen_expert_id = self.gating_network(input_tensor) 
        if chosen_expert_id in self.gpu_expert_cache: 
            print(f"Speculation HIT! Expert {chosen_expert_id} is already on GPU.") 
        else: 
            print(f"Speculation MISS! Loading expert {chosen_expert_id} now.") 
            self._load_expert_to_gpu(chosen_expert_id) 
        chosen_expert = self.gpu_expert_cache[chosen_expert_id] 
        output = chosen_expert(input_tensor) 
        self.expert_access_counts[chosen_expert_id] += 1 
        return output



# ------------------------- Not included in the book -------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    server = SpeculativeMoEServer(num_experts=8)

    for request_id in range(3):
        output = server.handle_request(torch.randn(1, 512))
        print(f"Request {request_id + 1} output shape: {tuple(output.shape)}")

    print("Demo complete. Cached experts:", sorted(server.gpu_expert_cache.keys()))
    print("Expert access counts:", server.expert_access_counts) 

# ------------------------- Not included in the book -------------------------