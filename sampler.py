# samplers.py
import math
import torch
from torch.utils.data import Sampler


class ShardSampler(Sampler[int]):
    def __init__(
        self,
        dataset_len: int,
        rank: int,
        world_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ):
        self.dataset_len = dataset_len
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        self.num_samples = dataset_len // world_size if drop_last else  math.ceil(dataset_len / world_size)
        self.total_size = self.num_samples * world_size

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        # 1) base indices
        indices = list(range(self.dataset_len))

        # 2) deterministic shuffle (seed + epoch)
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_len, generator = generator).tolist()

        # 3) pad or truncate to self.total_size
        if self.drop_last:
            indices = indices[:self.total_size]
        else:
            indices = indices + indices[:self.total_size - len(indices)] if len(indices) < self.total_size else indices

        # 4) shard by rank
        indices = indices[self.rank:self.total_size: self.world_size]

        assert len(indices) == self.num_samples
        
        for idx in indices:
            yield idx

    def __len__(self):
        # TODO: return num_samples
        return self.num_samples
