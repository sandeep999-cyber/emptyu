"""Masked Market Modeling — mask generator + grouped loss."""

from typing import Dict, Optional
import torch
import torch.nn.functional as F


class MaskGenerator:
    """Generates masks over valid data positions.

    CLS seam invariant: masks are drawn from the 512 data positions only.
    A single torch.Generator with fixed seed ensures determinism given
    the seeded EpochMarketSampler ordering.

    ``mask_mode``:
      - "random": random per-timestep selection (ratio of valid positions).
      - "span": contiguous spans of length ``span_len`` — forces prediction
        from surrounding context rather than trivially interpolating neighbors.
    """

    def __init__(
        self,
        mask_ratio: float = 0.15,
        seed: int = 42,
        mask_mode: str = "random",
        span_len: int = 16,
    ):
        self.mask_ratio = mask_ratio
        self.mask_mode = mask_mode
        self.span_len = max(1, int(span_len))
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def get_state(self):
        return self.generator.get_state()

    def set_state(self, state):
        self.generator.set_state(state)

    def __call__(self, mask: torch.Tensor) -> torch.Tensor:
        if self.mask_mode == "span":
            return self._span_mask(mask)
        return self._random_mask(mask)

    def _random_mask(self, mask: torch.Tensor) -> torch.Tensor:
        B, T = mask.shape
        # Only consider valid (True) data positions
        valid_count = mask.sum(dim=1)  # [B]
        target_count = (valid_count * self.mask_ratio).long().clamp(min=1)
        # Never mask more positions than are valid (handles ratio >= 1).
        target_count = target_count.clamp(max=valid_count)
        # Keep the gather index in bounds even for degenerate ratios.
        kth = target_count.clamp(max=max(0, T - 1)).unsqueeze(1)

        rand = torch.rand(B, T, generator=self.generator, device=mask.device)
        # Set invalid positions to 1 so they never get selected
        rand = rand.masked_fill(~mask, 1.0)
        threshold = rand.sort(dim=1).values.gather(1, kth)
        masked_positions = rand <= threshold
        # Clamp: never exceed valid count (edge case rounding)
        masked_positions = masked_positions & mask
        return masked_positions

    def _span_mask(self, mask: torch.Tensor) -> torch.Tensor:
        B, T = mask.shape
        masked = torch.zeros_like(mask)
        valid_count = mask.sum(dim=1)  # [B]
        n_spans = (
            (valid_count * self.mask_ratio) / max(1, self.span_len)
        ).long().clamp(min=1)
        n_spans = n_spans.clamp(max=valid_count.clamp(min=1))

        for b in range(B):
            vc = int(valid_count[b].item())
            if vc == 0:
                continue
            valid_idx = mask[b].nonzero(as_tuple=False).view(-1)
            ns = int(n_spans[b].item())
            # Stratified starts keep spans inside disjoint chunks so they never
            # overlap (clean contiguous blocks of at most span_len positions).
            chunk = max(1, vc // ns)
            off_hi = max(1, min(chunk, self.span_len))
            for k in range(ns):
                off = int(torch.randint(0, off_hi, (1,), generator=self.generator, device=mask.device).item())
                s = k * chunk + off
                hi = min(s + self.span_len, vc)
                masked[b][valid_idx[s:hi]] = True
        return masked


class MaskedMarketModelingLoss:
    """Compute per-group reconstruction loss over masked positions.

    Price group → Huber, funding/OI → MSE, calendar → per-field CE.
    Unavailable modalities are ignored via feature_mask + ignore_index.
    """

    def __init__(self, loss_cfg: Dict, d_model: int, device: torch.device):
        self.price_indices = loss_cfg["price_indices"]
        self.funding_oi_indices = loss_cfg["funding_oi_indices"]
        self.calendar_spec = loss_cfg.get("calendar") or {}
        self.reconstruct_calendar = bool(loss_cfg.get("reconstruct_calendar", True))
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

        # Calendar — per-field CE (input-only mode: skipped, see reconstruct_calendar)
        cal_total = 0.0
        cal_count = 0
        if self.reconstruct_calendar and self.calendar_spec:
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
