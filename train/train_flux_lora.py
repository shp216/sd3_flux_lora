"""LoRA fine-tuning for FLUX.1-dev (T2I) with inline text encoding.

Pipeline (per step):
  1. CLIP-L  -> pooled (B, 768)
  2. T5-XXL  -> sequence (B, S, 4096)         [both no_grad]
  3. VAE     -> latent (B, 16, H/8, W/8)      [no_grad]
  4. 2x2 pack latent -> (B, S_img, 64)
  5. Flow-matching timestep sample, mix noise+latent, predict velocity
  6. Loss = MSE(velocity_pred, noise - latent)  weighted by flow schedule

Only LoRA adapter weights are updated. Text encoders + VAE + base transformer are frozen.

Launch:
    accelerate launch --config_file accelerate_config.yaml train_flux_lora.py \
        --manifest /data/koreaai/Flux-Lora/train_data/manifest.jsonl \
        --output_dir /data/koreaai/Flux-Lora/ckpt/run1
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from bench_metrics import MetricEvaluator, load_fid_reference
from dataset import FluxLoraDataset, collate


# ----------------------- FLUX latent packing ----------------------- #
def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, H/2 * W/2, C*4)"""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(packed: torch.Tensor, h: int, w: int, c: int = 16) -> torch.Tensor:
    """(B, H/2 * W/2, C*4) -> (B, C, H, W)  -- reverse of pack_latents."""
    b = packed.shape[0]
    latents = packed.reshape(b, h // 2, w // 2, c, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, c, h, w)


def make_latent_image_ids(h: int, w: int, device, dtype) -> torch.Tensor:
    ids = torch.zeros(h // 2, w // 2, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(h // 2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(w // 2)[None, :]
    return ids.reshape(-1, 3).to(device=device, dtype=dtype)


# ------------------------- text encoding -------------------------- #
@torch.no_grad()
def encode_prompts(
    captions: list[str],
    clip_tok, clip, t5_tok, t5,
    device, dtype, t5_seq_len: int,
):
    """Returns (pooled_clip: (B, 768), t5_embed: (B, S, 4096))."""
    clip_in = clip_tok(captions, padding="max_length", max_length=77,
                       truncation=True, return_tensors="pt").to(device)
    pooled = clip(input_ids=clip_in.input_ids, output_hidden_states=False).pooler_output

    t5_in = t5_tok(captions, padding="max_length", max_length=t5_seq_len,
                   truncation=True, return_tensors="pt").to(device)
    t5_emb = t5(input_ids=t5_in.input_ids).last_hidden_state
    return pooled.to(dtype), t5_emb.to(dtype)


# ------------------------- training utils -------------------------- #
def get_sigmas(sched, timesteps, n_dim, device, dtype):
    sigmas = sched.sigmas.to(device=device, dtype=dtype)
    schedule_t = sched.timesteps.to(device)
    step_idx = [(schedule_t == t).nonzero().item() for t in timesteps.to(device)]
    sigma = sigmas[step_idx].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", default="black-forest-labs/FLUX.1-dev")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)

    # data / batch
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--t5_seq_len", type=int, default=128)

    # optim
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--lr_scheduler", default="constant_with_warmup")
    p.add_argument("--max_train_steps", type=int, default=50_000)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=64)

    # flow matching
    p.add_argument("--weighting_scheme", default="logit_normal",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--guidance_scale", type=float, default=1.0)

    # misc
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume_from", type=str, default=None)

    # wandb + eval (MMSVGBench text2svg: FID / CLIP / Aesthetic / HPS + grids)
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", default="flux-lora-omnisvg")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--eval_every", type=int, default=1000,
                   help="run full bench eval every N steps (0 = disable)")
    p.add_argument("--eval_on_start", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="run a full bench eval at step 0 before training")
    p.add_argument("--bench_parquet",
                   default="/data/shp216/Flux-Lora-train-bundle/train_data/raw/"
                           "MMSVGBench/data/text2svg-00000-of-00001.parquet")
    p.add_argument("--fid_ref_dir",
                   default="/data/shp216/Flux-Lora-train-bundle/train_data/fid_ref_white",
                   help="precomputed FID reference stats (build_fid_ref.py); "
                        "*_white = RGBA GT composited on white, which matches "
                        "the paper's numbers (raw convert('RGB') turns "
                        "transparent GT background black and inflates FID)")
    p.add_argument("--eval_trigger", default="",
                   help="optional prefix for bench prompts at generation time; "
                        "metrics always score the raw bench text")
    p.add_argument("--eval_resolution", type=int, default=1024)
    p.add_argument("--eval_inference_steps", type=int, default=28)
    p.add_argument("--eval_guidance_scale", type=float, default=3.5)
    p.add_argument("--grid_cell", type=int, default=256,
                   help="cell size (px) in the 10x10 wandb grids")
    return p.parse_args()


# OmniSVG 8B reference scores on MMSVGBench text2svg (first candidate,
# white-bg FID protocol) -- logged to wandb as constant series so training
# curves can be compared against them at a glance.
OMNISVG8B_BASELINE = {
    "fid_icon": 119.0612, "fid_illustration": 159.6584, "fid_mean": 139.3598,
    "clip_icon": 0.2741, "clip_illustration": 0.2132, "clip_mean": 0.2437,
    "aesthetic_icon": 4.5961, "aesthetic_illustration": 4.5233,
    "aesthetic_mean": 4.5597,
    "hps_icon": 0.2444, "hps_illustration": 0.2214, "hps_mean": 0.2329,
}


# ------------------------- eval utilities -------------------------- #
def load_bench(bench_parquet: str) -> dict:
    """MMSVGBench text2svg: 300 prompts (150 icon + 150 illustration)."""
    t = pq.read_table(bench_parquet, columns=["text", "type"])
    return {"texts": t.column("text").to_pylist(),
            "types": t.column("type").to_pylist()}


def make_grid(images: list[Image.Image], rows: int, cols: int,
              cell: int = 256) -> Image.Image:
    """Arrange images into rows x cols grid at `cell` px per side."""
    grid = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i in range(min(len(images), rows * cols)):
        r, c = divmod(i, cols)
        grid.paste(images[i].resize((cell, cell), Image.LANCZOS),
                   (c * cell, r * cell))
    return grid


def calculate_flux_shift(image_seq_len: int, base_seq_len: int = 256,
                          max_seq_len: int = 4096, base_shift: float = 0.5,
                          max_shift: float = 1.16) -> float:
    """FLUX-dev dynamic shift -- linear interp on sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@torch.no_grad()
def generate_one(
    prompt: str, seed: int, unwrapped, vae, clip, clip_tok, t5, t5_tok,
    sched_inference, res: int, num_inference_steps: int, guidance_scale: float,
    device, weight_dtype, t5_seq_len: int,
) -> Image.Image:
    """Manual FLUX denoise loop (no FluxPipeline) so dtypes stay consistent."""
    pooled, t5_emb = encode_prompts(
        [prompt], clip_tok, clip, t5_tok, t5,
        device=device, dtype=weight_dtype, t5_seq_len=t5_seq_len,
    )

    lh, lw = res // 8, res // 8
    gen = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn((1, 16, lh, lw), device=device, dtype=weight_dtype,
                         generator=gen)
    packed = pack_latents(latent)                        # (1, S_img, 64)
    img_ids = make_latent_image_ids(lh, lw, device, weight_dtype)
    txt_ids = torch.zeros(t5_emb.shape[1], 3, device=device, dtype=weight_dtype)

    mu = calculate_flux_shift(packed.shape[1])
    sched_inference.set_timesteps(num_inference_steps, device=device, mu=mu)

    for t_val in sched_inference.timesteps:
        t_in = t_val.expand(1).to(weight_dtype) / 1000.0
        guidance = torch.full([1], guidance_scale, device=device, dtype=weight_dtype)
        pred = unwrapped(
            hidden_states=packed,
            timestep=t_in,
            guidance=guidance,
            pooled_projections=pooled,
            encoder_hidden_states=t5_emb,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]
        packed = sched_inference.step(pred, t_val, packed, return_dict=False)[0]

    lat = unpack_latents(packed, lh, lw, c=16).to(weight_dtype)
    lat = (lat / vae.config.scaling_factor) + vae.config.shift_factor
    img = vae.decode(lat, return_dict=False)[0]
    img = (img / 2 + 0.5).clamp(0, 1)
    img = img[0].permute(1, 2, 0).float().cpu().numpy()
    return Image.fromarray((img * 255).astype(np.uint8))


def run_bench_eval(
    accel,
    transformer,
    vae, clip, clip_tok, t5, t5_tok,
    sched_inference,
    bench: dict,
    evaluator,           # MetricEvaluator | None (main process only)
    fid_ref: dict,
    step: int,
    args,
    weight_dtype: torch.dtype,
    out_dir: Path,
):
    """Full MMSVGBench text2svg eval, called on ALL ranks.

    Every rank generates its slice of the 300 prompts (trigger prepended);
    the main process then scores FID / CLIP / Aesthetic / HPS against the
    raw bench texts and logs three 10x10 grids + metrics to wandb.
    """
    device = accel.device
    unwrapped = accel.unwrap_model(transformer)
    was_training = unwrapped.training
    unwrapped.eval()

    texts = bench["texts"]
    types = bench["types"]
    trigger = args.eval_trigger.strip()
    gen_prompts = [f"{trigger} {t}" if trigger else t for t in texts]

    gen_dir = out_dir / "eval" / f"step{step:07d}" / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)

    try:
        my_indices = list(range(accel.process_index, len(texts), accel.num_processes))
        t0 = time.time()
        for k, idx in enumerate(my_indices):
            img = generate_one(
                gen_prompts[idx], seed=idx, unwrapped=unwrapped, vae=vae,
                clip=clip, clip_tok=clip_tok, t5=t5, t5_tok=t5_tok,
                sched_inference=sched_inference,
                res=args.eval_resolution,
                num_inference_steps=args.eval_inference_steps,
                guidance_scale=args.eval_guidance_scale,
                device=device, weight_dtype=weight_dtype,
                t5_seq_len=args.t5_seq_len,
            )
            img.save(gen_dir / f"{idx:03d}.png")
            if accel.is_main_process and (k + 1) % 10 == 0:
                done = (k + 1) * accel.num_processes
                print(f"[eval] step {step}: ~{done}/{len(texts)} generated "
                      f"({time.time() - t0:.0f}s)")
        accel.wait_for_everyone()

        if accel.is_main_process:
            paths = [str(gen_dir / f"{i:03d}.png") for i in range(len(texts))]
            missing = [p for p in paths if not Path(p).exists()]
            if missing:
                print(f"[eval] WARNING: {len(missing)} images missing, e.g. {missing[0]}")
                keep = [i for i, p in enumerate(paths) if Path(p).exists()]
                paths = [paths[i] for i in keep]
                texts_s = [texts[i] for i in keep]
                types_s = [types[i] for i in keep]
            else:
                texts_s, types_s = texts, types

            # metrics are scored against the RAW bench text (no trigger);
            # evaluate() also returns *_mean = (icon + illustration) / 2
            metrics = evaluator.evaluate(paths, texts_s, types_s, fid_ref)
            metrics_path = gen_dir.parent / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"step": step, **metrics}, f, indent=2)
            order = ["fid_mean", "clip_mean", "aesthetic_mean", "hps_mean",
                     "fid_icon", "fid_illustration"]
            summary = " | ".join(f"{k}={metrics[k]:.4f}" for k in order if k in metrics)
            print(f"[eval] step {step}: {summary}")

            # three 10x10 grids over the 300 prompts (bench order)
            grids = []
            imgs = [Image.open(p) for p in paths]
            for g in range(3):
                chunk = imgs[g * 100:(g + 1) * 100]
                if not chunk:
                    break
                grid = make_grid(chunk, rows=10, cols=10, cell=args.grid_cell)
                gp = gen_dir.parent / f"grid-{g}.jpg"
                grid.save(gp, quality=90)
                grids.append((gp, g))

            if args.use_wandb:
                import wandb
                log = {f"eval/{k}": v for k, v in metrics.items()}
                # constant OmniSVG-8B reference lines, one per eval metric
                log.update({f"eval/{k}_omnisvg8b": v
                            for k, v in OMNISVG8B_BASELINE.items()
                            if k in metrics})
                for gp, g in grids:
                    lo, hi = g * 100, min((g + 1) * 100, len(paths)) - 1
                    log[f"eval/grid_{g}"] = wandb.Image(
                        str(gp), caption=f"step {step} | prompts {lo}-{hi}")
                wandb.log(log, step=step)
        accel.wait_for_everyone()
    finally:
        if was_training:
            unwrapped.train()


# --------------------------------------------------------------- main #
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    accel = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=str(out_dir),
                                            logging_dir=str(out_dir / "logs")),
        # long NCCL timeout: main-process-only eval can stall other ranks well
        # past the default 10 min while they wait on the next allreduce
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=3))],
    )
    if accel.is_main_process:
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        if args.use_wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                dir=str(out_dir),
                config=vars(args),
                resume="allow",
            )

    set_seed(args.seed)
    weight_dtype = {"no": torch.float32, "fp16": torch.float16,
                    "bf16": torch.bfloat16}[args.mixed_precision]

    # ---------------- frozen components ---------------- #
    accel.print("Loading frozen modules: CLIP, T5, VAE, transformer ...")
    clip_tok = CLIPTokenizer.from_pretrained(args.pretrained_model, subfolder="tokenizer")
    clip = CLIPTextModel.from_pretrained(args.pretrained_model, subfolder="text_encoder",
                                          torch_dtype=weight_dtype)
    t5_tok = T5TokenizerFast.from_pretrained(args.pretrained_model, subfolder="tokenizer_2")
    t5 = T5EncoderModel.from_pretrained(args.pretrained_model, subfolder="text_encoder_2",
                                         torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model, subfolder="vae",
                                         torch_dtype=weight_dtype)
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model, subfolder="transformer", torch_dtype=weight_dtype)
    noise_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model, subfolder="scheduler")
    sched_train = copy.deepcopy(noise_sched)
    sched_eval = copy.deepcopy(noise_sched)  # separate copy for inference

    for m in (clip, t5, vae, transformer):
        m.requires_grad_(False)
        m.to(accel.device, dtype=weight_dtype).eval()

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.train()

    # ---------------- LoRA ---------------- #
    # exactly the default target set of diffusers' official
    # examples/dreambooth/train_dreambooth_lora_flux.py (for paper parity)
    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=[
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ],
    )
    transformer.add_adapter(lora_cfg)

    lora_params = []
    for _, p in transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
            lora_params.append(p)
    accel.print(f"Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    # ---------------- optim + data ---------------- #
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate,
                                   weight_decay=args.weight_decay)
    train_ds = FluxLoraDataset(args.manifest, resolution=args.resolution)
    accel.print(f"Dataset: {len(train_ds)} samples")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, collate_fn=collate,
                        drop_last=True, pin_memory=True)

    # prepared scheduler steps once per optimizer step on each process,
    # so scale by num_processes (not gradient_accumulation_steps)
    lr_sched = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accel.num_processes,
        num_training_steps=args.max_train_steps * accel.num_processes,
    )

    transformer, optimizer, loader, lr_sched = accel.prepare(
        transformer, optimizer, loader, lr_sched
    )

    vae_scale = vae.config.scaling_factor
    vae_shift = vae.config.shift_factor

    # ---------------- bench eval setup ---------------- #
    bench = None
    evaluator = None
    fid_ref = {}
    if args.eval_every > 0:
        bench = load_bench(args.bench_parquet)
        accel.print(f"Bench: {len(bench['texts'])} prompts from {args.bench_parquet}")
        if accel.is_main_process:
            evaluator = MetricEvaluator(device=str(accel.device))
            fid_ref = load_fid_reference(args.fid_ref_dir)
            if not fid_ref:
                accel.print(f"[warn] no FID reference stats in {args.fid_ref_dir} "
                            "-- run build_fid_ref.py first; FID will be skipped")
            with open(out_dir / "bench_prompts.json", "w") as f:
                json.dump(bench, f, indent=2, ensure_ascii=False)

    # ---------------- training loop ---------------- #
    global_step = 0
    if args.resume_from:
        global_step = load_checkpoint(
            accel, transformer, optimizer, lr_sched, args.resume_from,
        )

    last_eval_step = -1
    if args.eval_every > 0 and args.eval_on_start and global_step == 0:
        accel.print("[eval] step 0: evaluating base model before training ...")
        accel.wait_for_everyone()
        run_bench_eval(
            accel=accel, transformer=transformer,
            vae=vae, clip=clip, clip_tok=clip_tok, t5=t5, t5_tok=t5_tok,
            sched_inference=sched_eval, bench=bench, evaluator=evaluator,
            fid_ref=fid_ref, step=0, args=args,
            weight_dtype=weight_dtype, out_dir=out_dir,
        )
        last_eval_step = 0

    progress = tqdm(total=args.max_train_steps, disable=not accel.is_main_process,
                    initial=global_step, desc="train")
    t_start = time.time()
    running = 0.0
    running_n = 0
    epoch = 0

    while global_step < args.max_train_steps:
        epoch_completed = True
        for batch in loader:
            with accel.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(accel.device, dtype=weight_dtype,
                                                         non_blocking=True)
                captions = batch["captions"]
                bsz = pixel_values.shape[0]

                # ---- text encode (inline) ----
                pooled, t5_emb = encode_prompts(
                    captions, clip_tok, clip, t5_tok, t5,
                    device=accel.device, dtype=weight_dtype, t5_seq_len=args.t5_seq_len,
                )

                # ---- VAE encode ----
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae_shift) * vae_scale

                _, _, lh, lw = latents.shape
                packed = pack_latents(latents)
                img_ids = make_latent_image_ids(lh, lw, accel.device, weight_dtype)
                txt_ids = torch.zeros(t5_emb.shape[1], 3,
                                       device=accel.device, dtype=weight_dtype)

                # ---- sample timestep ----
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * sched_train.config.num_train_timesteps).long()
                timesteps = sched_train.timesteps[indices].to(accel.device)

                # ---- noise + interpolate ----
                noise = torch.randn_like(packed)
                sigmas = get_sigmas(sched_train, timesteps, n_dim=packed.ndim,
                                     device=accel.device, dtype=packed.dtype)
                noisy = (1.0 - sigmas) * packed + sigmas * noise

                # ---- forward ----
                guidance = torch.full([bsz], args.guidance_scale,
                                       device=accel.device, dtype=weight_dtype)
                model_pred = transformer(
                    hidden_states=noisy,
                    timestep=timesteps / 1000.0,
                    guidance=guidance,
                    pooled_projections=pooled,
                    encoder_hidden_states=t5_emb,
                    txt_ids=txt_ids,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]

                # ---- flow-matching loss ----
                target = noise - packed
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )
                loss = (weighting.float() * (model_pred.float() - target.float()) ** 2)
                loss = loss.reshape(loss.shape[0], -1).mean(1).mean()

                accel.backward(loss)
                if accel.sync_gradients:
                    accel.clip_grad_norm_(lora_params, args.max_grad_norm)
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            running += loss.detach().item()
            running_n += 1

            if accel.sync_gradients:
                global_step += 1
                progress.update(1)

                if global_step % args.log_every == 0:
                    avg = running / max(running_n, 1)
                    elapsed = time.time() - t_start
                    lr_now = lr_sched.get_last_lr()[0]
                    sps = global_step / elapsed
                    progress.set_postfix(loss=f"{avg:.4f}",
                                          lr=f"{lr_now:.1e}",
                                          steps_per_s=f"{sps:.2f}")
                    if accel.is_main_process and args.use_wandb:
                        import wandb
                        wandb.log({
                            "train/loss":          avg,
                            "train/lr":            lr_now,
                            "train/steps_per_sec": sps,
                            "train/step":          global_step,
                        }, step=global_step)
                    running = 0.0
                    running_n = 0

                if global_step % args.save_every == 0:
                    save_checkpoint(accel, transformer, optimizer, lr_sched,
                                     global_step, out_dir)

                if args.eval_every > 0 and global_step % args.eval_every == 0:
                    accel.wait_for_everyone()
                    run_bench_eval(
                        accel=accel,
                        transformer=transformer,
                        vae=vae, clip=clip, clip_tok=clip_tok,
                        t5=t5, t5_tok=t5_tok,
                        sched_inference=sched_eval,
                        bench=bench,
                        evaluator=evaluator,
                        fid_ref=fid_ref,
                        step=global_step,
                        args=args,
                        weight_dtype=weight_dtype,
                        out_dir=out_dir,
                    )
                    last_eval_step = global_step

                if global_step >= args.max_train_steps:
                    epoch_completed = False
                    break

        # ---- end of epoch: checkpoint + eval (skip if this step just ran one)
        if epoch_completed:
            epoch += 1
            accel.print(f"[epoch] epoch {epoch} finished at step {global_step}")
            save_checkpoint(accel, transformer, optimizer, lr_sched,
                            global_step, out_dir)
            if accel.is_main_process and args.use_wandb:
                import wandb
                wandb.log({"train/epoch": epoch}, step=global_step)
            if args.eval_every > 0 and global_step != last_eval_step:
                accel.wait_for_everyone()
                run_bench_eval(
                    accel=accel, transformer=transformer,
                    vae=vae, clip=clip, clip_tok=clip_tok, t5=t5, t5_tok=t5_tok,
                    sched_inference=sched_eval, bench=bench, evaluator=evaluator,
                    fid_ref=fid_ref, step=global_step, args=args,
                    weight_dtype=weight_dtype, out_dir=out_dir,
                )
                last_eval_step = global_step

    if accel.is_main_process:
        # save both the canonical step-tagged file (needed for resume) and a "final" alias
        _save_lora(accel.unwrap_model(transformer), out_dir, global_step)
        _save_lora(accel.unwrap_model(transformer), out_dir, "final")
        _save_resume_state(optimizer, lr_sched, global_step, out_dir)
        if args.use_wandb:
            import wandb
            wandb.finish()
    accel.print("Training done.")


def _save_lora(transformer, out_dir: Path, tag) -> None:
    """Inference-ready LoRA safetensors (loadable via FluxPipeline.load_lora_weights)."""
    state = get_peft_model_state_dict(transformer)
    prefixed = {f"transformer.{k}": v for k, v in state.items()}
    path = out_dir / f"flux_lora-{tag}.safetensors"
    save_file(prefixed, str(path))
    print(f"[save] {path} ({sum(v.numel() for v in state.values()):,} params)")


def _save_resume_state(optimizer, lr_sched, step: int, out_dir: Path) -> None:
    """Single resume_state.pt (overwrites previous) -- optimizer + scheduler + step."""
    path = out_dir / "resume_state.pt"
    torch.save({
        "step":         step,
        "optimizer":    optimizer.state_dict(),
        "lr_scheduler": lr_sched.state_dict(),
        "rng":          torch.get_rng_state(),
        "cuda_rng":     torch.cuda.get_rng_state_all(),
    }, path)
    print(f"[save] {path} (resume state @ step {step})")


def save_checkpoint(accel, transformer, optimizer, lr_sched, step: int, out_dir: Path) -> None:
    """Atomic combined save: LoRA safetensors + resume_state.pt. Main process only."""
    if not accel.is_main_process:
        return
    _save_lora(accel.unwrap_model(transformer), out_dir, step)
    _save_resume_state(optimizer, lr_sched, step, out_dir)


def load_checkpoint(accel, transformer, optimizer, lr_sched, resume_dir: str) -> int:
    """Restore LoRA + optimizer + scheduler from resume_dir/resume_state.pt.

    Returns the global step to continue from.
    """
    rd = Path(resume_dir)
    state_path = rd / "resume_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"resume_state.pt not found under {resume_dir}")

    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    step = int(ckpt["step"])
    lora_path = rd / f"flux_lora-{step}.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"matching LoRA weights not found: {lora_path}")

    # 1) load LoRA weights into transformer
    raw = load_file(str(lora_path))
    cleaned = {k.removeprefix("transformer."): v for k, v in raw.items()}
    unwrapped = accel.unwrap_model(transformer)
    set_peft_model_state_dict(unwrapped, cleaned)

    # 2) restore optimizer + LR schedule
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["lr_scheduler"])

    # 3) restore RNG (best-effort -- dataloader order will still differ unless seeded the same)
    try:
        torch.set_rng_state(ckpt["rng"])
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    except Exception as e:
        accel.print(f"[resume] RNG restore skipped: {e}")

    accel.print(f"[resume] step={step} | LoRA={lora_path.name}")
    return step


if __name__ == "__main__":
    main()
