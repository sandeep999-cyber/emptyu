"""Masked Market Modeling — mask generator + grouped loss."""

from typing import Dict, Optional
import torch
import torch.nn.functional as F


class MaskGenerator:
    """Generates 15% random timestep masks over valid data positions.

    CLS seam invariant: masks are drawn from the 512 data positions only.
    A single torch.Generator with fixed seed ensures determinism given
    the seeded EpochMarketSampler ordering.
    """

    def __init__(self, mask_ratio: float = 0.15, seed: int = 42):
        self.mask_ratio = mask_ratio
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def get_state(self):
        return self.generator.get_state()

    def set_state(self, state):
        self.generator.set_state(state)

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        B, T = mask.shape
        # Only consider valid (True) data positions
        valid_count = mask.sum(dim=1, keepdim=True)
        target_count = (valid_count * self.mask_ratio).long().clamp(min=1)

        rand = torch.rand(B, T, generator=self.generator, device=mask.device)
        # Set invalid positions to 1 so they never get selected
        rand = rand.masked_fill(~mask, 1.0)
        threshold = rand.sort(dim=1).values.gather(1, target_count)
        masked_positions = rand <= threshold
        # Clamp: never exceed valid count (edge case rounding)
        masked_positions = masked_positions & mask
        return masked_positions


class MaskedMarketModelingLoss:
    """Compute per-group reconstruction loss over masked positions.

    Price group → Huber, funding/OI → MSE, calendar → per-field CE.
    Unavailable modalities are ignored via feature_mask + ignore_index.
    """

    def __init__(self, loss_cfg: Dict, d_model: int, device: torch.device):
        self.price_indices = loss_cfg["price_indices"]
        self.funding_oi_indices = loss_cfg["funding_oi_indices"]
        self.calendar_spec = loss_cfg["calendar"]
        self.group_weights = loss_cfg.get("group_weights", {"price": 1.0, "funding_oi": 1.0, "calendar": 1.0})

    def __call__(
        self,
        reconstruction: Dict,
        features_norm: torch.Tensor,
        features_raw: torch.Tensor,
        feature_mask: torch.Tensor,
        masked_positions: torch.Tensor,
    ) -> Dict:
        """
        Args:
            reconstruction: dict with keys price [B,T,5], funding_oi [B,T,2],
                         calendar: {field: [B,T,classes]}
            features_norm: [B,T,15] normalized targets (for price and funding/OI)
            features_raw: [B,T,15] raw targets (for calendar CE)
            feature_mask: [B,T,15] bool — True = observed
            masked_positions: [B,T] bool — selected for masking
        Returns:
            dict with keys for each group loss + total_loss
        """
        mp = masked_positions.unsqueeze(-1)  # [B,T,1]
        losses = {}
        total = 0.0

        # Price group — Huber
        price_mask = mp * feature_mask[:, :, self.price_indices].all(dim=-1, keepdim=True)
        if price_mask.any():
            pred = reconstruction["price"]
            target = features_norm[:, :, self.price_indices]
            huber = F.huber_loss(pred, target, reduction="none", delta=1.0)
            losses["price"] = (huber * price_mask.float()).sum() / price_mask.sum().clamp(min=1)
            total += losses["price"] * self.group_weights.get("price", 1.0)

        # Funding/OI — MSE
        foi_mask = mp * feature_mask[:, :, self.funding_oi_indices].all(dim=-1, keepdim=True)
        if foi_mask.any():
            pred = reconstruction["funding_oi"]
            target = features_norm[:, :, self.funding_oi_indices]
            mse = F.mse_loss(pred, target, reduction="none")
            losses["funding_oi"] = (mse * foi_mask.float()).sum() / foi_mask.sum().clamp(min=1)
            total += losses["funding_oi"] * self.group_weights.get("funding_oi", 1.0)

        # Calendar — per-field CE
        cal_total = 0.0
        cal_count = 0
        for field, spec in self.calendar_spec.items():
            idx = spec["index"]
            classes = spec["classes"]
            offset = spec["offset"]
            field_mask = mp * feature_mask[:, :, idx:idx+1]  # [B,T,1]
            if field_mask.any():
                logits = reconstruction["calendar"][field]  # [B,T,classes]
                raw_target = features_raw[:, :, idx].long()
                if offset == 1:
                    target = raw_target - 1
                elif offset > 1:
                    target = raw_target - offset
                else:
                    target = raw_target
                # Clamp valid targets
                valid = (target >= 0) & (target < classes)
                target = target.masked_fill(~valid, -100)
                ce = F.cross_entropy(
                    logits.reshape(-1, classes),
                    target.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view_as(target)
                ce_masked = (ce * field_mask.squeeze(-1).float()).sum()
                count = field_mask.sum().clamp(min=1)
                if field not in losses:
                    losses[field] = ce_masked / count
                cal_total += losses[field]
                cal_count += 1

        if cal_count > 0:
            losses["calendar"] = cal_total / cal_count
            total += losses["calendar"] * self.group_weights.get("calendar", 1.0)

        losses["total"] = total
        return losses
