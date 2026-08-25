"""One-off smoke test: run all four metrics end-to-end on a few training
renders (also pre-downloads every metric model into the persistent cache)."""
import json
from pathlib import Path

from bench_metrics import MetricEvaluator, load_fid_reference

ROOT = Path("/data/shp216/Flux-Lora-train-bundle/train_data")

rows = []
with open(ROOT / "manifest.jsonl") as f:
    for line in f:
        rows.append(json.loads(line))
        if len(rows) >= 200:
            break
icon = [r for r in rows if r["src"] == "icon"][:6]
illu = [r for r in rows if r["src"] == "illu"][:6]
sample = icon + illu

paths = [str(ROOT / r["image"]) for r in sample]
texts = [r["caption"].replace("<vector>", "").strip() for r in sample]
types = ["icon" if r["src"] == "icon" else "illustration" for r in sample]

ev = MetricEvaluator()
fid_ref = load_fid_reference(ROOT / "fid_ref")
print("fid_ref keys:", list(fid_ref.keys()) or "(none yet -- FID skipped)")

res = ev.evaluate(paths, texts, types, fid_ref)
for k, v in sorted(res.items()):
    print(f"  {k}: {v:.4f}")
print("SMOKE_METRICS_OK")
