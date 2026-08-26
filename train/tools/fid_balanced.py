"""Balanced FID reference (N icon + N illustration, white-bg) and pooled FID of
each eval step's 300 generations (no icon/illu split) + OmniSVG 8B.
Usage: python tools/fid_balanced.py [N_per_type=10000]"""
import sys, json
sys.path.insert(0, "/data/shp216/Flux-Lora-train-bundle/train")
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from bench_metrics import MetricEvaluator, activation_statistics, frechet_distance
from build_fid_ref import iter_sampled_images

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
ROOT = Path("/data/shp216/Flux-Lora-train-bundle")
OUT = ROOT / "train_data" / f"fid_ref_bal{N//1000}k"; OUT.mkdir(parents=True, exist_ok=True)
ev = MetricEvaluator(); ev._ensure_loaded(); ev.inception.to(ev.device)

def feats_of_images(ims_iter, total):
    feats, batch = [], []
    for im in ims_iter:
        batch.append(ev.fid_tf(im))
        if len(batch) == 50:
            with torch.no_grad(): feats.append(ev.inception(torch.stack(batch).to(ev.device)).cpu())
            batch = []
    if batch:
        with torch.no_grad(): feats.append(ev.inception(torch.stack(batch).to(ev.device)).cpu())
    return torch.cat(feats).numpy()

# ---- 1) balanced reference ----
if not (OUT / "all.npz").exists():
    allf = []
    for key, repo in (("icon", "MMSVG-Icon"), ("illustration", "MMSVG-Illustration")):
        shards = sorted((ROOT / "train_data/raw" / repo / "data").glob("*.parquet"))
        total = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
        np.random.seed(42); idx = sorted(int(i) for i in np.random.choice(total, N, replace=False))
        f = feats_of_images(iter_sampled_images(shards, idx, "white"), N)
        mu, sg = activation_statistics(f); np.savez(OUT / f"{key}.npz", mu=mu, sigma=sg, n=len(f)); allf.append(f)
        print(f"[ref] {key}: {len(f)} images (of {total:,}, seed 42)", flush=True)
    f = np.concatenate(allf); mu, sg = activation_statistics(f)
    np.savez(OUT / "all.npz", mu=mu, sigma=sg, n=len(f)); print(f"[ref] all: {len(f)} -> {OUT}", flush=True)
ref = np.load(OUT / "all.npz"); mu_r, sg_r = ref["mu"], ref["sigma"]

def pooled_fid(paths):
    f = feats_of_images((Image.open(p).convert("RGB") for p in paths), len(paths))
    mu, sg = activation_statistics(f); return frechet_distance(mu_r, sg_r, mu, sg), len(paths)

# ---- 2) every eval step of the runs + OmniSVG ----
rows = []
for run in ("run-rank128-trigger-alllinear", "run-rank128-trigger"):
    for d in sorted((ROOT / "ckpt" / run / "eval").glob("step*/gen")):
        paths = sorted(d.glob("*.png"))
        if len(paths) < 290: continue
        step = int(d.parent.name[4:])
        old = json.load(open(d.parent / "metrics.json")).get("fid_mean", float("nan"))
        fid, n = pooled_fid(paths); rows.append((run, step, n, fid, old))
        print(f"{run:32s} step {step:5d} n={n} FID_bal{N//1000}k={fid:7.2f}  (old fid_mean={old:7.2f})", flush=True)
paths = sorted(p for t in ("icon", "illustration") for p in (ROOT / "omnisvg_repro/gen8b_final" / t).glob("*.png"))
fid, n = pooled_fid(paths); rows.append(("OmniSVG-8B", -1, n, fid, 139.36))
print(f"{'OmniSVG-8B':32s} first-cand n={n} FID_bal{N//1000}k={fid:7.2f}  (old fid_mean= 139.36)", flush=True)
json.dump([dict(run=r, step=s, n=n, fid_balanced=f, fid_mean_old=o) for r, s, n, f, o in rows],
          open(OUT / "results.json", "w"), indent=2)
print("BALANCED_FID_DONE")
