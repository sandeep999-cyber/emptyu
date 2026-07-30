"""Epoch-level sampler strategy."""

from typing import List, Optional
import numpy as np
import torch
from torch.utils.data import Sampler


class EpochMarketSampler(Sampler):
    """Custom Sampler for MarketDataset supporting epoch shuffling."""

    def __init__(self, data_source_len: int, shuffle: bool = True, seed: int = 42):
        self.data_source_len = data_source_len
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.data_source_len, generator=g).tolist()
        else:
            indices = list(range(self.data_source_len))
        return iter(indices)

    def __len__(self) -> int:
        return self.data_source_len
