"""Learning rate scheduler — linear warmup + cosine decay."""

import math
from typing import Any, Dict
from torch import optim


def build_scheduler(
    optimizer: optim.Optimizer,
    opt_cfg: Dict[str, Any],
    total_steps: int,
) -> optim.lr_scheduler.LambdaLR:
    sched_cfg = opt_cfg["scheduler"]
    warmup_steps = int(total_steps * sched_cfg["warmup_frac"])
    base_lr = opt_cfg["optimizer"]["adamw"]["lr"]
    lr_floor = sched_cfg.get("lr_floor", 0.0)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(progress * math.pi))
        return max(cosine, lr_floor / base_lr)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
