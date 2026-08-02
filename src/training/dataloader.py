"""PyTorch DataLoader builder."""

from typing import Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.data.market_dataset import MarketDataset
from src.training.sampler import EpochMarketSampler


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def create_dataloader(
    dataset: MarketDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    seed: int = 42,
    sampler: Optional[EpochMarketSampler] = None,
) -> DataLoader:
    """Create a PyTorch DataLoader over MarketDataset with seeded epoch shuffling."""
    worker_init = _seed_worker if num_workers > 0 else None
    if shuffle:
        if sampler is None:
            sampler = EpochMarketSampler(len(dataset), shuffle=True, seed=seed)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init,
            drop_last=True,
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init,
    )
