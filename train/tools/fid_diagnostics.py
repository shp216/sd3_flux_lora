"""FID calibration: what does FID look like for a PERFECT generator at n=150?

For each dataset and each background protocol (raw=black, white), sample 150
GT images that are NOT part of the reference set and compute their FID
against the reference. This is the small-sample bias floor: a generator
cannot score below ~this number under the same protocol.
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from tqdm import tqdm

from bench_metrics import (MetricEvaluator, activation_statistics,
                           frechet_distance, load_fid_reference)
from build_fid_ref import iter_sampled_images, sampled_indices

RAW = Path("/data/shp216/Flux-Lora-train-bundle/train_data/raw")
REFS = {
    "raw": Path("/data/shp216/Flux-Lora-train-bundle/train_data/fid_ref"),
    "white": Path("/data/shp216/Flux-Lora-train-bundle/train_data/fid_ref_white"),
}
HOLDOUT_N = 150

ev = MetricEvaluator()
ev._ensure_loaded()
ev.inception.to(ev.device)


def features_for(shards, indices, bg):
    feats, batch = [], []
    for img in tqdm(iter_sampled_images(shards, indices, bg), total=len(indices)):
        batch.append(ev.fid_tf(img))
        if len(batch) == 50:
            with torch.no_grad():
                feats.append(ev.inception(torch.stack(batch).to(ev.device)).cpu())
            batch = []
    if batch:
        with torch.no_grad():
            feats.append(ev.inception(torch.stack(batch).to(ev.device)).cpu())
    return torch.cat(feats).numpy()


for key, repo in (("icon", "MMSVG-Icon"), ("illustration", "MMSVG-Illustration")):
    shards = sorted((RAW / repo / "data").glob("*.parquet"))
    total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    ref_idx = set(sampled_indices(total, 0.03, 42))
    rng = np.random.default_rng(7)
    pool = [i for i in rng.choice(total, size=HOLDOUT_N * 3, replace=False)
            if i not in ref_idx][:HOLDOUT_N]
    pool = sorted(int(i) for i in pool)
    print(f"\n[{key}] holdout {len(pool)} GT images (disjoint from reference)")

    for bg, ref_dir in REFS.items():
        ref = load_fid_reference(ref_dir)
        if key not in ref:
            print(f"  bg={bg}: reference missing at {ref_dir}, skip")
            continue
        feats = features_for(shards, pool, bg)
        mu, sigma = activation_statistics(feats)
        fid = frechet_distance(ref[key]["mu"], ref[key]["sigma"], mu, sigma)
        print(f"  bg={bg:5s}: FID(GT-holdout-{HOLDOUT_N} vs ref) = {fid:.2f}")

print("\nFID_DIAG_OK")
