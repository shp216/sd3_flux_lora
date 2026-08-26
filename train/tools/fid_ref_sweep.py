"""Robustness sweep: FID of ours@step5000 (300) vs OmniSVG-8B (299) under many
reference-set constructions. All variants reported, favorable or not."""
import sys, json, random
sys.path.insert(0, "/data/shp216/Flux-Lora-train-bundle/train")
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
from PIL import Image
from bench_metrics import MetricEvaluator, activation_statistics, frechet_distance
from build_fid_ref import iter_sampled_images

ROOT = Path("/data/shp216/Flux-Lora-train-bundle"); RAW = ROOT / "train_data/raw"
CACHE = ROOT / "train_data/fid_ref_sweep"; CACHE.mkdir(exist_ok=True)
meta = json.load(open(ROOT / "omnisvg_repro/bench_meta.json")); types = meta["types"]
ev = MetricEvaluator(); ev._ensure_loaded(); ev.inception.to(ev.device)

def feats(ims_iter):
    out, b = [], []
    for im in ims_iter:
        b.append(ev.fid_tf(im))
        if len(b) == 50:
            with torch.no_grad(): out.append(ev.inception(torch.stack(b).to(ev.device)).cpu())
            b = []
    if b:
        with torch.no_grad(): out.append(ev.inception(torch.stack(b).to(ev.device)).cpu())
    return torch.cat(out).numpy()

SHARDS = {k: sorted((RAW / r / "data").glob("*.parquet")) for k, r in
          (("icon", "MMSVG-Icon"), ("illustration", "MMSVG-Illustration"))}
TOTAL = {k: sum(pq.ParquetFile(s).metadata.num_rows for s in v) for k, v in SHARDS.items()}

def ds_feats(key, n, seed):
    p = CACHE / f"{key}_{n}_s{seed}.npy"
    if p.exists(): return np.load(p)
    np.random.seed(seed); idx = sorted(int(i) for i in np.random.choice(TOTAL[key], n, replace=False))
    f = feats(iter_sampled_images(SHARDS[key], idx, "white")); np.save(p, f); return f

def trainrender_feats(n_per):
    p = CACHE / f"trainrender_{n_per}.npy"
    if p.exists(): return np.load(p)
    rows = [json.loads(l) for l in open(ROOT / "train_data/manifest.jsonl")]
    rng = random.Random(42); pick = []
    for src in ("icon", "illu"):
        pick += rng.sample([r for r in rows if r["src"] == src], n_per)
    f = feats(Image.open(ROOT / "train_data" / r["image"]).convert("RGB") for r in pick); np.save(p, f); return f

# generated sets
ours = [ROOT / f"ckpt/run-rank128-trigger-alllinear/eval/step0005000/gen/{i:03d}.png" for i in range(300)]
omni = sorted(p for t in ("icon", "illustration") for p in (ROOT / "omnisvg_repro/gen8b_final" / t).glob("*.png"))
G = {"ours@5000": feats(Image.open(p).convert("RGB") for p in ours),
     "ours@5000->512": feats(Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS) for p in ours),
     "OmniSVG-8B": feats(Image.open(p).convert("RGB") for p in omni)}
omni_types = [types[int(p.stem)] for p in omni]

def fid(ref_f, gen_f):
    m1, s1 = activation_statistics(ref_f); m2, s2 = activation_statistics(gen_f); return frechet_distance(m1, s1, m2, s2)
def split_fid(ref_icon, ref_illu, gen_f, gtypes):
    mi = np.array([t == "icon" for t in gtypes])
    return (fid(ref_icon, gen_f[mi]) + fid(ref_illu, gen_f[~mi])) / 2

print(f"{'reference variant':38s} {'ours@5000':>10s} {'ours->512':>10s} {'OmniSVG':>9s}   winner", flush=True)
def row(name, ref_f=None, split=None):
    if split is not None:
        vals = [split_fid(split[0], split[1], G[k], types if k != "OmniSVG-8B" else omni_types) for k in G]
    else:
        vals = [fid(ref_f, G[k]) for k in G]
    win = "OURS" if vals[0] < vals[2] else "omni"
    print(f"{name:38s} {vals[0]:10.1f} {vals[1]:10.1f} {vals[2]:9.1f}   {win}", flush=True)

# existing 3% protocol (features not cached -> recompute via indices file)
import gzip
fr = json.load(gzip.open(ROOT / "data/dataset_manifest_compact.json.gz", "rt"))["fid_reference"]
def pct3_feats(key):
    p = CACHE / f"{key}_3pct_s42.npy"
    if p.exists(): return np.load(p)
    f = feats(iter_sampled_images(SHARDS[key], fr[key]["row_indices_in_load_dataset_order"], "white")); np.save(p, f); return f
ic3, il3 = pct3_feats("icon"), pct3_feats("illustration")
row("3% seed42, split (current protocol)", split=(ic3, il3))
row("3% seed42, pooled 34,782", ref_f=np.concatenate([ic3, il3]))
for seed in (42, 0, 1):
    ic, il = ds_feats("icon", 10000, seed), ds_feats("illustration", 10000, seed)
    row(f"balanced 10k+10k seed{seed}, pooled", ref_f=np.concatenate([ic, il]))
    if seed == 42:
        row("balanced 10k+10k seed42, split", split=(ic, il))
        row("icon-only 10k", ref_f=ic); row("illustration-only 10k", ref_f=il)
for n in (1000, 3000):
    row(f"balanced {n}+{n} seed42, pooled", ref_f=np.concatenate([ds_feats("icon", n, 42), ds_feats("illustration", n, 42)]))
tr = trainrender_feats(5000)
row("our cairosvg-1024 train renders 5k+5k", ref_f=tr)
print("SWEEP_DONE")
