"""Pick, per bench prompt, the candidate with the highest CLIP score.

Input: omnisvg_repro/<gen_root>/chunk_*/NNNN_<name>_candidate_K.png
Output: omnisvg_repro/<out_root>/icon/{gi:03d}.png and .../illustration/{gi:03d}.png
        (exactly one image per prompt, 150 icon + 150 illustration)

CLIP scoring matches OmniSVG compute_clip.py: openai/clip-vit-base-patch32,
cosine similarity between image and the raw bench text.

Usage: python select_best_clip.py gen4b_all best4b
"""
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

REPRO = Path("/data/shp216/Flux-Lora-train-bundle/omnisvg_repro")
gen_root = REPRO / sys.argv[1]
out_root = REPRO / sys.argv[2]

meta = json.load(open(REPRO / "bench_meta.json"))
texts, types = meta["texts"], meta["types"]

device = "cuda"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


@torch.no_grad()
def clip_scores(paths: list[Path], text: str) -> list[float]:
    imgs = [Image.open(p).convert("RGB") for p in paths]
    inputs = proc(text=[text], images=imgs, return_tensors="pt",
                  padding=True, truncation=True).to(device)
    img_f = model.get_image_features(pixel_values=inputs["pixel_values"])
    txt_f = model.get_text_features(input_ids=inputs["input_ids"],
                                    attention_mask=inputs["attention_mask"])
    if not torch.is_tensor(img_f):
        img_f = img_f.pooler_output
    if not torch.is_tensor(txt_f):
        txt_f = txt_f.pooler_output
    img_f = img_f / img_f.norm(dim=-1, keepdim=True)
    txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
    return (img_f @ txt_f.T).squeeze(-1).cpu().tolist()


for t in ("icon", "illustration"):
    (out_root / t).mkdir(parents=True, exist_ok=True)

picked, missing = 0, []
for ch in meta["chunks"]:
    d = gen_root / f"chunk_{ch['chunk']}"
    groups: dict[int, list[Path]] = defaultdict(list)
    for p in sorted(d.glob("*.png")):
        m = re.match(r"(\d{4})_", p.name)
        if m:
            groups[int(m.group(1)) - 1].append(p)
    for local in range(ch["hi"] - ch["lo"]):
        gi = ch["lo"] + local
        cands = groups.get(local, [])
        if not cands:
            missing.append(gi)
            continue
        scores = clip_scores(cands, texts[gi])
        best = cands[max(range(len(cands)), key=lambda k: scores[k])]
        shutil.copy(best, out_root / types[gi] / f"{gi:03d}.png")
        picked += 1

n_icon = len(list((out_root / "icon").glob("*.png")))
n_illu = len(list((out_root / "illustration").glob("*.png")))
print(f"[{sys.argv[1]}] picked {picked} best-of-CLIP images "
      f"(icon {n_icon}, illustration {n_illu}) | missing prompts: {missing}")
print("SELECT_BEST_OK")
