"""AdamW optimizer factory from optimizer_v1.yaml config."""

from typing import Dict, Any
import torch
from torch import optim


def build_optimizer(model: torch.nn.Module, opt_cfg: Dict[str, Any]) -> optim.Optimizer:
    adamw_cfg = opt_cfg["optimizer"]["adamw"]
    return optim.AdamW(
        model.parameters(),
        lr=adamw_cfg["lr"],
        weight_decay=adamw_cfg["weight_decay"],
        betas=tuple(adamw_cfg["betas"]),
        eps=adamw_cfg["eps"],
    )
