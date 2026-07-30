"""PyTorch DataLoader builder."""

from typing import Optional
import torch
from torch.utils.data import DataLoader
from src.data.market_dataset import MarketDataset
from src.training.sampler import EpochMarketSampler


def create_dataloader(
    dataset: MarketDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Create a PyTorch DataLoader over MarketDataset with seeded epoch shuffling."""
    if shuffle:
        sampler = EpochMarketSampler(len(dataset), shuffle=True, seed=seed)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
