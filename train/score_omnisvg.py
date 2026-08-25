"""Score OmniSVG 4B generations on MMSVGBench text2svg with the same metric
pipeline used during FLUX-LoRA training, and compare against the paper.

Paper (OmniSVG 3B/4B, text2svg):
                 FID     CLIP   Aesthetic  HPS
  icon         137.40   0.275     4.62    0.244
  illustration 154.37   0.226     4.56    0.232
"""
import json
import re
import sys
from pathlib import Path

from bench_metrics import MetricEvaluator, load_fid_reference

ROOT = Path("/data/shp216/Flux-Lora-train-bundle")
REPRO = ROOT / "omnisvg_repro"
REF_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "train_data" / "fid_ref"
GEN_NAME = sys.argv[2] if len(sys.argv) > 2 else "gen"  # e.g. "gen" (4B) or "gen8b"

PAPER = {
    "icon":         {"fid": 137.40, "clip": 0.275, "aesthetic": 4.62, "hps": 0.244},
    "illustration": {"fid": 154.37, "clip": 0.226, "aesthetic": 4.56, "hps": 0.232},
}

meta = json.load(open(REPRO / "bench_meta.json"))
texts, types = meta["texts"], meta["types"]

paths: list[str] = []
keep_texts: list[str] = []
keep_types: list[str] = []
missing = []
for ch in meta["chunks"]:
    d = REPRO / GEN_NAME / f"chunk_{ch['chunk']}"
    found = {}
    for p in d.glob("*.png"):
        m = re.match(r"(\d{4})_", p.name)
        if m:
            found[int(m.group(1)) - 1] = p  # inference.py numbers from 1
    for local in range(ch["hi"] - ch["lo"]):
        gi = ch["lo"] + local
        if local in found:
            paths.append(str(found[local]))
            keep_texts.append(texts[gi])
            keep_types.append(types[gi])
        else:
            missing.append(gi)

n_icon = sum(1 for t in keep_types if t == "icon")
n_illu = len(keep_types) - n_icon
print(f"scored images: {len(paths)} (icon {n_icon}, illustration {n_illu}) "
      f"| missing {len(missing)}: {missing[:10]}{'...' if len(missing) > 10 else ''}")

print(f"FID reference: {REF_DIR}")
ev = MetricEvaluator()
fid_ref = load_fid_reference(REF_DIR)
res = ev.evaluate(paths, keep_texts, keep_types, fid_ref)

json.dump(res, open(REPRO / f"metrics-{GEN_NAME}-{REF_DIR.name}.json", "w"), indent=2)

print(f"\n{'':14s}{'FID':>10s}{'CLIP':>9s}{'Aesthetic':>11s}{'HPS':>9s}")
for t in ("icon", "illustration"):
    ours = (res.get(f"fid_{t}", float('nan')), res.get(f"clip_{t}", float('nan')),
            res.get(f"aesthetic_{t}", float('nan')), res.get(f"hps_{t}", float('nan')))
    pap = PAPER[t]
    print(f"{t:14s}{ours[0]:10.2f}{ours[1]:9.3f}{ours[2]:11.2f}{ours[3]:9.3f}   <- ours")
    print(f"{'':14s}{pap['fid']:10.2f}{pap['clip']:9.3f}{pap['aesthetic']:11.2f}{pap['hps']:9.3f}   <- paper")
print(f"\nall: fid={res.get('fid_all', float('nan')):.2f} clip={res['clip']:.3f} "
      f"aesthetic={res['aesthetic']:.2f} hps={res['hps']:.3f}")
print("SCORE_OMNISVG_OK")
