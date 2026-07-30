"""Gather final stats for report."""
import sys, json
from pathlib import Path
sys.path.insert(0, ".")
from src.data.db import db_manager

snap = Path("D:/emptyu/storage/training/snapshots/2026-07-30")
print("SNAPSHOT CONTENTS:")
for f in snap.iterdir():
    print(f"  {f.name}: {f.stat().st_size} bytes")

with open(snap / "checksums.json") as f:
    cs = json.load(f)
print(f"  files in checksums: {cs['file_count']}")

with open(snap / "content_hash") as f:
    print(f"  content_hash: {f.read().strip()[:32]}...")

raw = list(Path("D:/emptyu/storage/raw").rglob("*.zip"))
canon = list(Path("D:/emptyu/storage/canonical").rglob("*.parquet"))
raw_gb = sum(f.stat().st_size for f in raw) / 1e9
canon_gb = sum(f.stat().st_size for f in canon) / 1e9
print(f"\nSTORAGE:")
print(f"  raw: {len(raw)} zips, {raw_gb:.2f} GB")
print(f"  canonical: {len(canon)} parquets, {canon_gb:.2f} GB")

files = db_manager.query_files()
total_rows = sum(f["row_count"] for f in files)
print(f"  index: {len(files)} files, {total_rows:,} total rows")

by_type = {}
for f in files:
    k = f["dataset_type"]
    by_type.setdefault(k, [0, 0])
    by_type[k][0] += 1
    by_type[k][1] += f["row_count"]
print("\nBY TYPE:")
for k, (n, r) in sorted(by_type.items()):
    print(f"  {k}: {n} files, {r:,} rows")
