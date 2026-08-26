"""Is the MMSVG FID reference biased toward white/blank/monochrome images?
(1) describe the GT reference distribution (ink coverage, saturation, mono ratio)
(2) FID sensitivity: same generated set, with background whitened / desaturated /
    both / replaced by pure white -- how much does FID move without any change in
    content quality?"""
import sys, json, io, gzip
sys.path.insert(0, "/data/shp216/Flux-Lora-train-bundle/train")
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from bench_metrics import MetricEvaluator, activation_statistics, frechet_distance, load_fid_reference
from build_fid_ref import iter_sampled_images

ROOT = Path("/data/shp216/Flux-Lora-train-bundle")
GEN = ROOT / "ckpt/run-rank128-trigger/eval/step0001000/gen"      # FID_mean 179.6 set
meta = json.load(open(ROOT / "omnisvg_repro/bench_meta.json")); types = meta["types"]
ref = load_fid_reference(ROOT / "train_data/fid_ref_white")
ev = MetricEvaluator(); ev._ensure_loaded(); ev.inception.to(ev.device)

def img_stats(im):
    a = np.asarray(im.convert("RGB").resize((256, 256))).astype(np.float32)
    ink = (a.min(-1) < 235).mean()                       # non-white pixel fraction
    sat = np.asarray(im.convert("RGB").resize((128,128)).convert("HSV"))[..., 1].mean() / 255
    return ink, sat

# ---------- (1) reference distribution ----------
print("=== (1) FID reference GT distribution (1,000 sampled per type, white-composited) ===")
with gzip.open(ROOT / "data/dataset_manifest_compact.json.gz", "rt") as f:
    fr = json.load(f)["fid_reference"]
for key, repo in (("icon", "MMSVG-Icon"), ("illustration", "MMSVG-Illustration")):
    shards = sorted((ROOT / "train_data/raw" / repo / "data").glob("*.parquet"))
    idx = fr[key]["row_indices_in_load_dataset_order"]
    rng = np.random.default_rng(0); pick = sorted(int(i) for i in rng.choice(idx, 1000, replace=False))
    inks, sats = [], []
    for im in iter_sampled_images(shards, pick, "white"):
        i, s = img_stats(im); inks.append(i); sats.append(s)
    inks, sats = np.array(inks), np.array(sats)
    print(f"  {key:13s} ink coverage: mean {inks.mean():.1%}  median {np.median(inks):.1%}  "
          f"p10 {np.percentile(inks,10):.1%} p90 {np.percentile(inks,90):.1%} | "
          f"white-space >70%: {(inks<0.3).mean():.0%} | mono(sat<0.08): {(sats<0.08).mean():.0%} | sat {sats.mean():.3f}")

# ---------- (2) FID sensitivity on a fixed generated set ----------
def whiten_bg(im, tol=40):
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = np.median(border, axis=0)
    if bg.mean() > 235: return im
    mask = (np.abs(a - bg).max(-1) < tol)
    a[mask] = 255
    return Image.fromarray(a.astype(np.uint8))
def gray(im): return im.convert("L").convert("RGB")
def white(im): return Image.new("RGB", im.size, (255, 255, 255))

variants = {"original": lambda im: im, "bg->white": whiten_bg, "grayscale": gray,
            "bg->white + gray": lambda im: gray(whiten_bg(im)), "pure white x300": white}
paths = [GEN / f"{i:03d}.png" for i in range(300)]
print("\n=== (2) FID of the SAME step-1000 generations under content-preserving edits ===")
print(f"{'variant':20s} {'fid_icon':>9s} {'fid_illu':>9s} {'fid_mean':>9s}   ink   sat")
for name, fn in variants.items():
    ims = [fn(Image.open(p)) for p in paths]
    st = np.array([img_stats(im) for im in ims])
    feats = []
    for i in range(0, 300, 50):
        b = torch.stack([ev.fid_tf(im) for im in ims[i:i+50]]).to(ev.device)
        with torch.no_grad(): feats.append(ev.inception(b).cpu())
    feats = torch.cat(feats).numpy()
    out = {}
    for t in ("icon", "illustration"):
        m = np.array([tt == t for tt in types])
        mu, sg = activation_statistics(feats[m]); out[t] = frechet_distance(ref[t]["mu"], ref[t]["sigma"], mu, sg)
    print(f"{name:20s} {out['icon']:9.1f} {out['illustration']:9.1f} {(out['icon']+out['illustration'])/2:9.1f}   {st[:,0].mean():.0%}  {st[:,1].mean():.2f}")
print("PROBE_DONE")
