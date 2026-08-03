"""AdamW optimizer factory from optimizer_v1.yaml config."""

from typing import Dict, Any
import torch
from torch import optim


def build_optimizer(model: torch.nn.Module, opt_cfg: Dict[str, Any]) -> optim.Optimizer:
    adamw_cfg = opt_cfg["optimizer"]["adamw"]
    # Fused kernel (CUDA) merges the AdamW update into a single launch; on CPU
    # the foreach multi-tensor path is the faster fallback. Both preserve the
    # optimizer semantics (same effective batch, LR curve, weight decay).
    fused = bool(adamw_cfg.get("fused", torch.cuda.is_available()))
    kwargs: Dict[str, Any] = {
        "lr": adamw_cfg["lr"],
        "weight_decay": adamw_cfg["weight_decay"],
        "betas": tuple(adamw_cfg["betas"]),
        "eps": adamw_cfg["eps"],
    }
    if fused and torch.cuda.is_available():
        kwargs["fused"] = True
    else:
        kwargs["foreach"] = True
    return optim.AdamW(model.parameters(), **kwargs)
