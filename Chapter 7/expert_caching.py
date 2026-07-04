import torch 
from collections import OrderedDict 

class ExpertCache: 
    def __init__(self, capacity, device): 
        self.cache = OrderedDict() 
        self.capacity = capacity 
        self.device = device 

    def get(self, expert_id): 
        if expert_id not in self.cache: 
            return None 
        # Move the accessed expert to the end of the OrderedDict 
        self.cache.move_to_end(expert_id) 
        return self.cache[expert_id] 

    def put(self, expert_id, expert_weights): 
        if expert_id in self.cache: 
            # Update the existing expert 
            self.cache[expert_id] = expert_weights.to(self.device) 
            self.cache.move_to_end(expert_id) 
        else: 
            if len(self.cache) >= self.capacity: 
                # Evict the least recently used expert 
                lru_expert_id, _ = self.cache.popitem(last=False) 
                print(f"Evicting expert {lru_expert_id} from cache.") 
            # Add the new expert 
            self.cache[expert_id] = expert_weights.to(self.device) 

expert_dim = 1024 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cache = ExpertCache(capacity=2, device=device)

expert_0 = torch.randn(expert_dim, device=device)
expert_1 = torch.randn(expert_dim, device=device)
expert_2 = torch.randn(expert_dim, device=device)

cache.put(expert_id=0, expert_weights=expert_0) 
cache.put(expert_id=1, expert_weights=expert_1) 

cached_expert = cache.get(expert_id=0) 
if cached_expert is not None: 
    print("Expert 0 retrieved from cache") 

cache.put(expert_id=2, expert_weights=expert_2) 
assert cache.get(1) is None 