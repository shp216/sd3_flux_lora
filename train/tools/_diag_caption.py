"""Diagnostic: is the LoRA style caption-conditional?
A = raw bench prompt (eval protocol)   B = training-caption-style wrapper
C = base FLUX (no LoRA) + wrapper.  All: step-4500 LoRA unless C."""
import json, torch, numpy as np
from pathlib import Path
from PIL import Image
from diffusers import FluxPipeline

ROOT = Path("/data/shp216/Flux-Lora-train-bundle")
OUT = ROOT / "ckpt/diag_caption"
LORA = ROOT / "ckpt/run-rank128/flux_lora-4500.safetensors"

rows = [json.loads(l) for l in open(ROOT / "train_data/eval_prompts.jsonl")]
prompts = [(r["caption"].replace("<vector>", "").strip(), r["type"]) for r in rows]

def wrap(p, t):
    kind = "flat vector icon" if t == "icon" else "flat vector illustration"
    core = p[0].lower() + p[1:].rstrip(".")
    return f"The image features a {kind} of {core}, with simple shapes and solid colors, set against a white background."

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

def run(tag, texts, lora):
    if lora:
        pipe.load_lora_weights(str(LORA.parent), weight_name=LORA.name, adapter_name="d")
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
    imgs = []
    for i, t in enumerate(texts):
        g = torch.Generator("cuda").manual_seed(i)
        im = pipe(t, height=1024, width=1024, num_inference_steps=28,
                  guidance_scale=3.5, generator=g).images[0]
        im.save(d / f"{i:02d}.png"); imgs.append(im)
    if lora:
        pipe.unload_lora_weights()
    grid = Image.new("RGB", (5*256, 4*256), "white")
    for i, im in enumerate(imgs):
        grid.paste(im.resize((256, 256)), ((i % 5)*256, (i // 5)*256))
    grid.save(OUT / f"grid_{tag}.jpg", quality=90)
    # stats
    wb = dk = 0; sat = []
    for im in imgs:
        a = np.asarray(im.resize((128,128))).astype(np.float32)
        b = np.concatenate([a[0],a[-1],a[:,0],a[:,-1]]).mean()
        wb += b > 235; dk += b < 80
        sat.append(np.asarray(im.convert("HSV").resize((128,128)))[...,1].mean()/255)
    print(f"[{tag}] whiteBG={wb}/20 darkBG={dk}/20 sat={np.mean(sat):.3f}", flush=True)

raw = [p for p, _ in prompts]
wrapped = [wrap(p, t) for p, t in prompts]
run("A_raw_lora", raw, lora=True)
run("B_wrapped_lora", wrapped, lora=True)
run("C_wrapped_base", wrapped, lora=False)
print("DIAG_DONE")
