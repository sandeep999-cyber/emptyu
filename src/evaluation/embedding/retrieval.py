"""Nearest-neighbor retrieval evaluation on pooled embeddings.

Measures whether similar market states have nearby embeddings:
cosine kNN same-symbol hit rate and cross-symbol retrieval.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def evaluate_retrieval(emb_data: dict, k: int = 10) -> dict:
    embeddings = emb_data["embedding"]
    symbols = np.array(emb_data["symbols"])
    if len(embeddings) < k + 1:
        return {"error": f"Too few samples ({len(embeddings)}) for k={k}"}

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)

    same_symbol = 0
    total = 0
    for i, neighs in enumerate(indices):
        for j in neighs[1:]:
            if symbols[i] == symbols[j]:
                same_symbol += 1
            total += 1
    same_symbol_rate = same_symbol / total if total > 0 else 0.0
    cross_symbol_rate = 1.0 - same_symbol_rate

    return {
        "k": k,
        "n_samples": len(embeddings),
        "same_symbol_rate": round(same_symbol_rate, 4),
        "cross_symbol_rate": round(cross_symbol_rate, 4),
        "valid": True,
    }


def main():
    from src.evaluation.embedding._common import load_model_and_normalizer, extract_split_embeddings

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Run dir from CheckpointManager")
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean", "attention"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation"])
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, configs = load_model_and_normalizer(Path(args.checkpoint), device)
    emb_data = extract_split_embeddings(
        model, normalizer, args.split, args.pooling,
        configs["trainer_config"], device, max_windows=args.max_windows,
    )
    results = evaluate_retrieval(emb_data, k=args.k)

    out_dir = Path("evaluation/embedding")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"retrieval_{args.split}_{args.pooling}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Retrieval results written to {path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
