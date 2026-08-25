"""Precompute FID reference statistics for MMSVGBench evaluation.

Replicates OmniSVG's protocol (compute_fid.download_gt_dataset): from each of
MMSVG-Icon / MMSVG-Illustration take np.random.seed(42) + np.random.choice
3% of rows (the dataset's own 896x896 `image` renders), run them through
torchvision InceptionV3 (fc removed), and store mu/sigma. Also stores a
combined 'all' reference from the concatenated features.

Run once (inside the flux-lora container, any GPU with ~2GB free):
    python build_fid_ref.py \
        --raw_root /data/shp216/Flux-Lora-train-bundle/train_data/raw \
        --out_dir  /data/shp216/Flux-Lora-train-bundle/train_data/fid_ref
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from tqdm import tqdm

from bench_metrics import MetricEvaluator, activation_statistics


def sampled_indices(total: int, pct: float, seed: int) -> list[int]:
    np.random.seed(seed)
    idx = np.random.choice(total, size=int(total * pct), replace=False)
    return sorted(int(i) for i in idx)


def to_rgb(img: Image.Image, bg: str) -> Image.Image:
    """bg='raw': plain convert('RGB') -- alpha dropped, transparent pixels go
    black (what OmniSVG's compute_fid literally does). bg='white': composite
    RGBA onto white first (matches how generated SVGs are rendered)."""
    if bg == "white" and img.mode in ("RGBA", "LA", "PA"):
        img = img.convert("RGBA")
        base = Image.new("RGBA", img.size, (255, 255, 255, 255))
        return Image.alpha_composite(base, img).convert("RGB")
    return img.convert("RGB")


def iter_sampled_images(shards: list[Path], indices: list[int], bg: str):
    """Yield PIL images for global row indices over concatenated shards
    (shards sorted by filename = datasets.load_dataset row order)."""
    counts = [pq.ParquetFile(s).metadata.num_rows for s in shards]
    offsets = np.cumsum([0] + counts)
    by_shard: dict[int, list[int]] = {}
    for gi in indices:
        s = int(np.searchsorted(offsets, gi, side="right") - 1)
        by_shard.setdefault(s, []).append(gi - int(offsets[s]))
    for s in sorted(by_shard):
        table = pq.read_table(shards[s], columns=["image"])
        col = table.column("image")
        for li in by_shard[s]:
            img = col[li].as_py()
            data = img["bytes"] if isinstance(img, dict) else img
            yield to_rgb(Image.open(io.BytesIO(data)), bg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--pct", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=50)
    ap.add_argument("--bg", choices=["raw", "white"], default="raw",
                    help="'raw' = alpha dropped (transparent -> black, literal "
                         "OmniSVG code path); 'white' = composite onto white")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ev = MetricEvaluator(batch_size=args.batch_size)
    ev._ensure_loaded()
    ev.inception.to(ev.device)

    all_feats = []
    for key, repo in (("icon", "MMSVG-Icon"), ("illustration", "MMSVG-Illustration")):
        shards = sorted((args.raw_root / repo / "data").glob("*.parquet"))
        total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
        idx = sampled_indices(total, args.pct, args.seed)
        print(f"[{key}] {total:,} rows -> {len(idx):,} reference images")

        feats, batch = [], []
        for img in tqdm(iter_sampled_images(shards, idx, args.bg), total=len(idx), desc=key):
            batch.append(ev.fid_tf(img))
            if len(batch) == args.batch_size:
                with torch.no_grad():
                    feats.append(ev.inception(torch.stack(batch).to(ev.device)).cpu())
                batch = []
        if batch:
            with torch.no_grad():
                feats.append(ev.inception(torch.stack(batch).to(ev.device)).cpu())
        feats = torch.cat(feats).numpy()
        all_feats.append(feats)

        mu, sigma = activation_statistics(feats)
        np.savez(args.out_dir / f"{key}.npz", mu=mu, sigma=sigma, n=len(feats))
        print(f"[{key}] saved stats over {feats.shape[0]} images")

    combined = np.concatenate(all_feats, axis=0)
    mu, sigma = activation_statistics(combined)
    np.savez(args.out_dir / "all.npz", mu=mu, sigma=sigma, n=len(combined))
    print(f"[all] saved stats over {combined.shape[0]} images -> {args.out_dir}")


if __name__ == "__main__":
    main()
