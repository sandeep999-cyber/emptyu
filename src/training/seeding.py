"""Global RNG seeding for reproducible runs."""

import random
import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch RNGs (plus CUDA) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
