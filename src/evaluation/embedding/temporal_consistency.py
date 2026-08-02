"""Temporal consistency of embedding evolution.

Measures whether adjacent market states produce smooth embedding
transitions: cosine similarity of temporally adjacent same-symbol
windows vs random pairs, with separation AUC.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def evaluate_temporal_consistency(emb_data: dict) -> dict:
    embeddings = emb_data["embedding"]
    symbols = np.array(emb_data["symbols"])
    start_ms = np.array(emb_data["window_start_ms"])
    n = len(embeddings)
    if n < 100:
        return {"error": f"Too few samples ({n})"}

    # Sort chronologically; adjacency only within same symbol
    order = np.argsort(start_ms, kind="stable")
    embeddings = embeddings[order]
    symbols = symbols[order]

    def _cos(a, b):
        return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))

    adj_cos = np.array([
        _cos(embeddings[i], embeddings[i + 1])
        for i in range(n - 1)
        if symbols[i] == symbols[i + 1]
    ])

    rng = np.random.default_rng(42)
    rand_cos = []
    attempts = 0
    while len(rand_cos) < min(10000, n * 10) and attempts < 100000:
        i, j = rng.integers(0, n, size=2)
        attempts += 1
        if i == j or symbols[i] != symbols[j]:
            continue
        rand_cos.append(_cos(embeddings[i], embeddings[j]))
    rand_cos = np.array(rand_cos)

    if len(rand_cos) > 0 and len(adj_cos) > 1:
        y_true = np.concatenate([np.ones_like(adj_cos), np.zeros_like(rand_cos)])
        y_score = np.concatenate([adj_cos, rand_cos])
        auc = float(roc_auc_score(y_true, y_score))
    else:
        auc = 0.5

    return {
        "n_samples": n,
        "adjacent_mean_cos": round(float(adj_cos.mean()), 4) if len(adj_cos) else None,
        "adjacent_std_cos": round(float(adj_cos.std()), 4) if len(adj_cos) else None,
        "random_mean_cos": round(float(rand_cos.mean()), 4) if len(rand_cos) else None,
        "separation_auc": round(auc, 4),
        "adjacent_pairs": len(adj_cos),
        "random_pairs": len(rand_cos),
    }


def main():
    from src.evaluation.embedding._common import load_model_and_normalizer, extract_split_embeddings

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Run dir from CheckpointManager")
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean", "attention"])
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, configs = load_model_and_normalizer(Path(args.checkpoint), device)
    emb_data = extract_split_embeddings(
        model, normalizer, args.split, args.pooling,
        configs["trainer_config"], device, max_windows=args.max_windows,
    )
    results = evaluate_temporal_consistency(emb_data)

    out_dir = Path("evaluation/embedding")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"temporal_{args.split}_{args.pooling}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Temporal consistency results written to {path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
