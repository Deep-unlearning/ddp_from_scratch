"""Check how PyTorch's DistributedSampler shards data"""

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, size=20):
        self.size = size
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return idx

dataset = DummyDataset(20)

print("PyTorch DistributedSampler (world_size=2, shuffle=False):\n")

for rank in range(2):
    sampler = DistributedSampler(
        dataset, 
        num_replicas=2, 
        rank=rank, 
        shuffle=False,
        drop_last=True
    )
    indices = list(sampler)
    print(f"Rank {rank}: {indices}")

print("\n" + "="*50)
print("\nYour ShardSampler (world_size=2, shuffle=False):\n")

import sys
sys.path.insert(0, '/home/user/ddp_from_scratch')
from sampler import ShardSampler

for rank in range(2):
    sampler = ShardSampler(
        dataset_len=20,
        rank=rank,
        world_size=2,
        shuffle=False,
        drop_last=True
    )
    indices = list(sampler)
    print(f"Rank {rank}: {indices}")
