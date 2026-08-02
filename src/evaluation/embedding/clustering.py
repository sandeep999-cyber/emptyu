"""Embedding clustering evaluation — KMeans on pooled embeddings.

Measures whether embeddings naturally separate market regimes
(trending/ranging, high/low volatility, etc.) using silhouette score
and Adjusted Mutual Information vs derived regime labels.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_mutual_info_score


def _derive_regime_labels(emb_data: dict) -> np.ndarray:
    """Assign pseudo-labels from embedding-norm median split."""
    norms = np.linalg.norm(emb_data["embedding"], axis=1)
    median = np.median(norms)
    return (norms >= median).astype(np.int32)


def evaluate_clustering(emb_data: dict, n_clusters: int = 8, max_silhouette_samples: int = 8000) -> dict:
    """Cluster pooled embeddings with KMeans and score with silhouette/AMI.

    KMeans runs on the full embedding set, but silhouette/AMI are computed
    on a seeded subsample when the dataset is large: silhouette_score builds
    a dense N x N distance matrix, so full-size runs on e.g. 65k train
    windows would require ~34 GB and crash the runtime.
    """
    embeddings = emb_data["embedding"]
    n = len(embeddings)
    if n < n_clusters:
        return {"error": f"Too few samples ({n}) for {n_clusters} clusters"}

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    sil_emb = embeddings
    sil_labels = labels
    sil_idx = np.arange(n)
    if n > max_silhouette_samples:
        rng = np.random.default_rng(0)
        sil_idx = rng.choice(n, max_silhouette_samples, replace=False)
        sil_emb = embeddings[sil_idx]
        sil_labels = labels[sil_idx]
    sil = float(silhouette_score(sil_emb, sil_labels))

    pseudo_labels = _derive_regime_labels(emb_data)
    ami = float(adjusted_mutual_info_score(pseudo_labels[sil_idx], sil_labels))

    cluster_counts = {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}

    return {
        "n_clusters": n_clusters,
        "n_samples": n,
        "n_silhouette_samples": int(len(sil_emb)),
        "silhouette": round(sil, 4),
        "ami_vs_norm_regime": round(ami, 4),
        "inertia": float(kmeans.inertia_),
        "cluster_sizes": cluster_counts,
    }


def main():
    from src.evaluation.embedding._common import load_model_and_normalizer, extract_split_embeddings

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Run dir from CheckpointManager")
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean", "attention"])
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-windows", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, normalizer, configs = load_model_and_normalizer(Path(args.checkpoint), device)
    emb_data = extract_split_embeddings(
        model, normalizer, args.split, args.pooling,
        configs["trainer_config"], device, max_windows=args.max_windows,
    )
    results = evaluate_clustering(emb_data, n_clusters=args.clusters)

    out_dir = Path("evaluation/embedding")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"clustering_{args.split}_{args.pooling}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Clustering results written to {path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
