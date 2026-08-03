"""Visualization: PCA / t-SNE of embeddings and training loss curves."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_2d_projections(emb_data: dict, title: str, out_path: Path, method: str = "pca"):
    embeddings = emb_data["embedding"]
    symbols = emb_data["symbols"]

    if len(embeddings) > 5000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(embeddings), size=5000, replace=False)
        embeddings = embeddings[idx]
        symbols = [symbols[i] for i in idx]

    if method == "tsne":
        proj = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(embeddings)
    else:
        proj = PCA(n_components=2, random_state=42).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(10, 8))
    for sym in sorted(set(symbols)):
        mask = np.array([s == sym for s in symbols])
        ax.scatter(proj[mask, 0], proj[mask, 1], label=sym, alpha=0.5, s=5)
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(run_dir: Path, out_path: Path):
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest.json found in {run_dir}")
        return
    manifest = json.loads(manifest_path.read_text())
    checkpoints = manifest.get("checkpoints", [])
    if not checkpoints:
        print("No checkpoint history in manifest.")
        return

    epochs = [c["epoch"] for c in checkpoints]
    train_losses = [c.get("train_loss") for c in checkpoints]
    val_losses = [c.get("val_loss") for c in checkpoints]

    fig, ax = plt.subplots(figsize=(10, 6))
    if any(tl is not None for tl in train_losses):
        ax.plot(epochs, train_losses, label="Train", marker="o")
    if any(vl is not None for vl in val_losses):
        ax.plot(epochs, val_losses, label="Validation", marker="s")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()
    ax.grid(True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    from src.evaluation.embedding._common import load_model_and_normalizer, extract_split_embeddings

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Run dir from CheckpointManager")
    parser.add_argument("--pooling", type=str, default="mean", choices=["cls", "mean", "attention"])
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "tsne"])
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.checkpoint)
    model, normalizer, configs = load_model_and_normalizer(run_dir, device)

    out_dir = Path("evaluation/embedding/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        emb_data = extract_split_embeddings(
            model, normalizer, split, args.pooling,
            configs["trainer_config"], device, max_windows=args.max_windows,
            batch_size=args.batch_size,
        )
        if len(emb_data["embedding"]) == 0:
            continue
        path = out_dir / f"{args.method}_{split}_{args.pooling}.png"
        plot_2d_projections(emb_data, f"{args.method.upper()} — {split} ({args.pooling})", path, method=args.method)
        print(f"Saved {path}")

    loss_path = out_dir / "loss_curves.png"
    plot_loss_curves(run_dir, loss_path)
    print(f"Saved {loss_path}")


if __name__ == "__main__":
    main()
