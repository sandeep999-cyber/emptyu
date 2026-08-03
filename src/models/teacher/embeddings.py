"""Embedding pooling strategies and EmbeddingExtractor API.

Three pooling modes over the encoder's per-timestep latent:
  - cls: output at the CLS position
  - mean: mask-aware mean over data positions
  - attention: learned query attending over data positions (mask-aware)

EmbeddingExtractor: loads a trained TeacherEncoder + normalizer from a
checkpoint directory and produces embeddings for any MarketDataset.
"""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def pool_cls(latent: torch.Tensor) -> torch.Tensor:
    return latent[:, 0, :]


def pool_mean(latent: torch.Tensor, key_padding_mask: torch.Tensor, t_data: int) -> torch.Tensor:
    data_latents = latent[:, 1:, :]  # [B, T_data, D]
    data_mask = key_padding_mask[:, 1:]  # [B, T_data]
    data_mask_3d = data_mask.unsqueeze(-1).float()
    masked_sum = (data_latents * data_mask_3d).sum(dim=1)
    count = data_mask_3d.sum(dim=1).clamp(min=1)
    return masked_sum / count


class AttentionPooling(nn.Module):
    """Learned query over data positions (mask-aware).

    Note: this layer is never trained by the current training loop; it is
    initialized with a fixed seed so that extraction is deterministic, but
    the pooled output reflects the untrained (random) query rather than a
    learned attention. Prefer ``cls`` or ``mean`` for research pooling.
    """

    def __init__(self, d_model: int, seed: int = 42):
        super().__init__()
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.query = nn.Parameter(torch.randn(1, 1, d_model, generator=gen) * 0.02)
        self.linear = nn.Linear(d_model, d_model)
        # Use the same private generator for the projection. The previous
        # default initialization consumed global RNG state, making attention
        # pooling differ between separate extraction processes.
        bound = d_model ** -0.5
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound, generator=gen)
            self.linear.bias.zero_()

    def forward(self, latent: torch.Tensor, key_padding_mask: torch.Tensor, t_data: int) -> torch.Tensor:
        data_latents = latent[:, 1:, :]  # [B, T_data, D]
        data_mask = key_padding_mask[:, 1:]  # [B, T_data]

        q = self.query.expand(latent.shape[0], -1, -1)  # [B, 1, D]
        k = self.linear(data_latents)  # [B, T_data, D]
        scores = torch.bmm(q, k.transpose(1, 2)) / (latent.shape[-1] ** 0.5)  # [B, 1, T_data]

        # Mask: set padded positions to -inf
        scores = scores.masked_fill(~data_mask.unsqueeze(1), float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        pooled = torch.bmm(attn_weights, data_latents).squeeze(1)
        return pooled


def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    pooling: str,
    device: torch.device,
) -> Dict:
    """Extract pooled embeddings for all samples in a dataloader.

    Returns dict with keys:
      embedding: np.ndarray [N, d_model]
      symbols: list of str
      window_start_ms: list of int
      window_end_ms: list of int
    """
    model.eval()
    embeddings = []
    symbols = []
    start_tss = []
    end_tss = []

    attn_pool = AttentionPooling(model.d_model).to(device) if pooling == "attention" else None
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            timestamps = batch["timestamps"].to(device)
            mask = batch["mask"].to(device)

            latent, kpm, positions, t_data = model(features, timestamps, mask)

            if pooling == "cls":
                pooled = pool_cls(latent)
            elif pooling == "mean":
                pooled = pool_mean(latent, kpm, t_data)
            elif pooling == "attention":
                pooled = attn_pool(latent, kpm, t_data)
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            embeddings.append(pooled.cpu().numpy())
            meta = batch.get("metadata")
            n = features.shape[0]
            if isinstance(meta, dict):
                # Default collate converts list-of-per-sample-dicts into
                # dict-of-columns (str -> list, int -> tensor). Un-collate.
                meta = [
                    {
                        k: (v[i].item() if hasattr(v[i], "item") else v[i])
                        for k, v in meta.items()
                    }
                    for i in range(n)
                ]
            elif meta is None:
                meta = [{}] * n
            for m in meta:
                symbols.append(m.get("symbol", "unknown"))
                start_tss.append(m.get("window_start_ms", 0))
                end_tss.append(m.get("window_end_ms", 0))

    return {
        "embedding": np.concatenate(embeddings, axis=0),
        "symbols": symbols,
        "window_start_ms": start_tss,
        "window_end_ms": end_tss,
    }


def extract_embeddings_multi(
    model: nn.Module,
    dataloader: DataLoader,
    poolings: List[str],
    device: torch.device,
) -> Dict[str, Dict]:
    """Extract several pooling modes while running the encoder once per batch.

    Pooling only changes the reduction of the encoder latent sequence. Running
    the transformer once and applying all reductions avoids three full forward
    passes when comparing cls/mean/attention.
    """
    invalid = set(poolings) - {"cls", "mean", "attention"}
    if invalid:
        raise ValueError(f"Unknown pooling mode(s): {sorted(invalid)}")

    model.eval()
    outputs = {
        pooling: {"embedding": [], "symbols": [], "window_start_ms": [], "window_end_ms": []}
        for pooling in poolings
    }
    attn_pool = AttentionPooling(model.d_model).to(device) if "attention" in poolings else None

    # Keep the same no-grad execution mode as extract_embeddings so callers
    # get numerically identical results across the two APIs.
    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            timestamps = batch["timestamps"].to(device)
            mask = batch["mask"].to(device)
            latent, kpm, positions, t_data = model(features, timestamps, mask)

            pooled = {"cls": pool_cls(latent)} if "cls" in poolings else {}
            if "mean" in poolings:
                pooled["mean"] = pool_mean(latent, kpm, t_data)
            if "attention" in poolings:
                pooled["attention"] = attn_pool(latent, kpm, t_data)

            meta = batch.get("metadata")
            n = features.shape[0]
            if isinstance(meta, dict):
                meta = [
                    {
                        k: (v[i].item() if hasattr(v[i], "item") else v[i])
                        for k, v in meta.items()
                    }
                    for i in range(n)
                ]
            elif meta is None:
                meta = [{}] * n

            for pooling in poolings:
                outputs[pooling]["embedding"].append(pooled[pooling].cpu().numpy())
                outputs[pooling]["symbols"].extend(m.get("symbol", "unknown") for m in meta)
                outputs[pooling]["window_start_ms"].extend(
                    m.get("window_start_ms", 0) for m in meta
                )
                outputs[pooling]["window_end_ms"].extend(
                    m.get("window_end_ms", 0) for m in meta
                )

    for output in outputs.values():
        output["embedding"] = np.concatenate(output["embedding"], axis=0)
    return outputs
