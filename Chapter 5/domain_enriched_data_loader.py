import random 
from collections import defaultdict 

class DomainEnrichedDataLoader: 
    def __init__(self, data, domain_labels, batch_size=32, skew_ratio=0.7): 
        self.data = data 
        self.batch_size = batch_size 
        self.skew_ratio = skew_ratio 

        self.num_batches = len(self.data) // self.batch_size

        self.domain_indices = defaultdict(list) 

        for i, label in enumerate(domain_labels): 
            self.domain_indices[label].append(i) 
    
        self.all_indices = list(range(len(data))) 
        self.domains = list(self.domain_indices.keys()) 
    
    def __iter__(self): 
        self.num_batches = len(self.data) // self.batch_size 
        return self 

    def __next__(self): 
        if self.num_batches <= 0: 
            raise StopIteration 
        
        primary_domain = random.choice(self.domains) 

        primary_samples_count = int(self.batch_size * self.skew_ratio) 
        background_samples_count = self.batch_size - primary_samples_count 

        all_primary_indices = self.domain_indices[primary_domain] 
        primary_indices = random.sample( all_primary_indices, min(primary_samples_count, len(all_primary_indices)) ) 

        background_pool = [i for i in self.all_indices if i not in all_primary_indices] 
        background_indices = random.sample(background_pool, min(background_samples_count, len(background_pool))) 
        
        batch_indices = primary_indices + background_indices 
        random.shuffle(batch_indices) 
        self.num_batches -= 1 
        return [self.data[i] for i in batch_indices] 
    
all_docs = [f"Doc {i}" for i in range(200)] 
domain_labels = (['code']*100) + (['legal']*50) + (['science']*50) 

data_loader = DomainEnrichedDataLoader(all_docs, domain_labels, batch_size=10, skew_ratio=0.7) 
first_batch = next(data_loader) 
print(f"Generated a batch of size: {len(first_batch)}") 
print(first_batch)

#for batch in data_loader: 
#    model.train(batch) 