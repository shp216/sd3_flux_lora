"""LoRA fine-tuning for Stable Diffusion 3 / 3.5 (T2I) -- SD3 twin of train_flux_lora.py.

Same data pipeline, bench evaluation (MMSVGBench text2svg: FID/CLIP/Aesthetic/HPS
+ 10x10 grids + wandb with OmniSVG-8B reference lines), checkpointing and resume
logic as the FLUX script. SD3-specific parts follow diffusers'
examples/dreambooth/train_dreambooth_lora_sd3.py:

  * text: CLIP-L + CLIP-G (penultimate hidden states, concat -> zero-pad to 4096)
          ++ T5-XXL sequence; pooled = concat(CLIP-L, CLIP-G projections)
  * latents: VAE (B,16,H/8,W/8), (z - shift) * scale, NO 2x2 packing
  * timesteps passed as 0..1000 (not /1000), true CFG at inference (guidance 7.0)
  * LoRA targets = official default (attention only)
  * precondition_outputs=1 (official default): pred*(-sigma)+noisy vs latents

Launch (inside the docker container):
    accelerate launch --num_processes 8 --mixed_precision bf16 train_sd3_lora.py \
        --manifest .../train_data/manifest.jsonl --output_dir .../ckpt/sd3-run1 \
        --eval_trigger "<vector>" --use_wandb
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
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
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
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from bench_metrics import MetricEvaluator, load_fid_reference
from dataset import FluxLoraDataset, collate

# OmniSVG 8B reference scores (MMSVGBench text2svg, first candidate, white-bg FID)
OMNISVG8B_BASELINE = {
    "fid_icon": 119.0612, "fid_illustration": 159.6584, "fid_mean": 139.3598,
    "clip_icon": 0.2741, "clip_illustration": 0.2132, "clip_mean": 0.2437,
    "aesthetic_icon": 4.5961, "aesthetic_illustration": 4.5233,
    "aesthetic_mean": 4.5597,
    "hps_icon": 0.2444, "hps_illustration": 0.2214, "hps_mean": 0.2329,
}


# ------------------------- LoRA target sets -------------------------- #
# full-match regexes over module names (embedders / norm_out / top-level
# proj_out are never matched). SD3.5-medium (MMDiT-X) also has `attn2`
# (dual attention) in the first blocks -> covered by `attn2?`.
_BLK = r"transformer_blocks\.\d+\."
_ATTN = r"attn2?\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)"
LORA_TARGETS = {
    # diffusers examples/dreambooth/train_dreambooth_lora_sd3.py default (attention only)
    "official": _BLK + r"attn\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out)",
    # kohya-style: every Linear inside the blocks (attn, attn2, FFN, AdaLN modulation)
    "all-linear": _BLK + r"(" + _ATTN
                  + r"|ff\.net\.(0\.proj|2)|ff_context\.net\.(0\.proj|2)|norm1\.linear|norm1_context\.linear)",
    # attention (incl. attn2) + feed-forward, NO modulation -- shared paper placement
    "attn-ffn": _BLK + r"(" + _ATTN + r"|ff\.net\.(0\.proj|2)|ff_context\.net\.(0\.proj|2))",
}


# ------------------------- text encoding (SD3) -------------------------- #
@torch.no_grad()
def _encode_clip(text_encoder, tokenizer, captions, device):
    ids = tokenizer(captions, padding="max_length", max_length=77,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    out = text_encoder(ids, output_hidden_states=True)
    return out.hidden_states[-2], out[0]          # (B,77,D), pooled (B,D)


@torch.no_grad()
def encode_prompts_sd3(captions, clip_tok, clip_l, clip_tok2, clip_g,
                       t5_tok, t5, device, dtype, t5_seq_len):
    """Returns (prompt_embeds (B, 77+S, 4096), pooled (B, 2048)) exactly as
    diffusers' SD3 pipeline/training encode_prompt does."""
    h1, p1 = _encode_clip(clip_l, clip_tok, captions, device)
    h2, p2 = _encode_clip(clip_g, clip_tok2, captions, device)
    clip_h = torch.cat([h1, h2], dim=-1)                       # (B,77,2048)
    t5_ids = t5_tok(captions, padding="max_length", max_length=t5_seq_len,
                    truncation=True, return_tensors="pt").input_ids.to(device)
    t5_h = t5(t5_ids)[0]                                       # (B,S,4096)
    clip_h = F.pad(clip_h, (0, t5_h.shape[-1] - clip_h.shape[-1]))
    prompt_embeds = torch.cat([clip_h, t5_h], dim=1)
    pooled = torch.cat([p1, p2], dim=-1)
    return prompt_embeds.to(dtype), pooled.to(dtype)


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
    p.add_argument("--pretrained_model",
                   default="stabilityai/stable-diffusion-3.5-medium",
                   help="SD3 / SD3.5 diffusers repo (e.g. stabilityai/"
                        "stable-diffusion-3-medium-diffusers)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output_dir", required=True)

    # data / batch
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--t5_seq_len", type=int, default=77,
                   help="official SD3 LoRA script default is 77")

    # optim
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--lr_warmup_steps", type=int, default=200)
    p.add_argument("--lr_scheduler", default="constant_with_warmup")
    p.add_argument("--max_train_steps", type=int, default=15_000)

    # LoRA
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lora_targets", default="all-linear", choices=sorted(LORA_TARGETS),
                   help="'all-linear' = all Linear in blocks (kohya style); "
                        "'official' = diffusers SD3 example default (attention only)")

    # flow matching (same as FLUX script / official SD3 script)
    p.add_argument("--weighting_scheme", default="logit_normal",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--precondition_outputs", type=int, default=1,
                   help="official SD3 script default 1: x0-style target")

    # misc
    p.add_argument("--mixed_precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--init_lora", type=str, default=None,
                   help="LoRA safetensors to load as initialization (weights only, "
                        "fresh optimizer)")
    p.add_argument("--start_step", type=int, default=0,
                   help="global step to count from when using --init_lora")

    # wandb + eval (identical protocol to the FLUX script)
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", default="flux-lora-omnisvg")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--eval_on_start", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--bench_parquet",
                   default="/data/shp216/Flux-Lora-train-bundle/train_data/raw/"
                           "MMSVGBench/data/text2svg-00000-of-00001.parquet")
    p.add_argument("--fid_ref_dir",
                   default="/data/shp216/Flux-Lora-train-bundle/train_data/fid_ref_white")
    p.add_argument("--eval_trigger", default="",
                   help="prefix for bench prompts at generation time only; "
                        "metrics always score the raw bench text")
    p.add_argument("--eval_resolution", type=int, default=1024)
    p.add_argument("--eval_inference_steps", type=int, default=28)
    p.add_argument("--eval_guidance_scale", type=float, default=7.0,
                   help="SD3 pipeline default (true CFG)")
    p.add_argument("--grid_cell", type=int, default=256)
    return p.parse_args()


# ------------------------- eval utilities -------------------------- #
def load_bench(bench_parquet: str) -> dict:
    t = pq.read_table(bench_parquet, columns=["text", "type"])
    return {"texts": t.column("text").to_pylist(),
            "types": t.column("type").to_pylist()}


def make_grid(images, rows, cols, cell=256):
    grid = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    for i in range(min(len(images), rows * cols)):
        r, c = divmod(i, cols)
        grid.paste(images[i].resize((cell, cell), Image.LANCZOS), (c * cell, r * cell))
    return grid


def run_bench_eval(accel, pipe, bench, evaluator, fid_ref, step, args, out_dir,
                   weight_dtype=torch.bfloat16):
    """All ranks generate their slice with the SD3 pipeline (LoRA-attached
    transformer); main process scores + logs. Same protocol as FLUX."""
    unwrapped = pipe.transformer
    was_training = unwrapped.training
    unwrapped.eval()
    texts, types = bench["texts"], bench["types"]
    trigger = args.eval_trigger.strip()
    gen_prompts = [f"{trigger} {t}" if trigger else t for t in texts]
    gen_dir = out_dir / "eval" / f"step{step:07d}" / "gen"
    gen_dir.mkdir(parents=True, exist_ok=True)
    try:
        my = list(range(accel.process_index, len(texts), accel.num_processes))
        t0 = time.time()
        for k, idx in enumerate(my):
            g = torch.Generator(device=accel.device).manual_seed(idx)
            # autocast: fp32 LoRA layers promote the transformer output to
            # fp32, and the SD3 pipeline feeds that straight into the bf16 VAE
            with torch.autocast("cuda", dtype=weight_dtype):
                img = pipe(gen_prompts[idx], height=args.eval_resolution,
                           width=args.eval_resolution,
                           num_inference_steps=args.eval_inference_steps,
                           guidance_scale=args.eval_guidance_scale,
                           max_sequence_length=args.t5_seq_len,
                           generator=g).images[0]
            img.save(gen_dir / f"{idx:03d}.png")
            if accel.is_main_process and (k + 1) % 10 == 0:
                print(f"[eval] step {step}: ~{(k+1)*accel.num_processes}/{len(texts)} "
                      f"generated ({time.time()-t0:.0f}s)")
        accel.wait_for_everyone()

        if accel.is_main_process:
            paths = [str(gen_dir / f"{i:03d}.png") for i in range(len(texts))]
            keep = [i for i, p in enumerate(paths) if Path(p).exists()]
            paths = [paths[i] for i in keep]
            texts_s = [texts[i] for i in keep]
            types_s = [types[i] for i in keep]
            metrics = evaluator.evaluate(paths, texts_s, types_s, fid_ref)
            with open(gen_dir.parent / "metrics.json", "w") as f:
                json.dump({"step": step, **metrics}, f, indent=2)
            order = ["fid_mean", "clip_mean", "aesthetic_mean", "hps_mean",
                     "fid_icon", "fid_illustration"]
            print(f"[eval] step {step}: " +
                  " | ".join(f"{k}={metrics[k]:.4f}" for k in order if k in metrics))
            grids = []
            imgs = [Image.open(p) for p in paths]
            for gi in range(3):
                chunk = imgs[gi*100:(gi+1)*100]
                if not chunk:
                    break
                gp = gen_dir.parent / f"grid-{gi}.jpg"
                make_grid(chunk, 10, 10, args.grid_cell).save(gp, quality=90)
                grids.append((gp, gi))
            if args.use_wandb:
                import wandb
                log = {f"eval/{k}": v for k, v in metrics.items()}
                log.update({f"eval/{k}_omnisvg8b": v for k, v in
                            OMNISVG8B_BASELINE.items() if k in metrics})
                for gp, gi in grids:
                    lo, hi = gi*100, min((gi+1)*100, len(paths)) - 1
                    log[f"eval/grid_{gi}"] = wandb.Image(
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
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=3))],
    )
    if accel.is_main_process:
        with open(out_dir / "args.json", "w") as f:
            json.dump(vars(args), f, indent=2)
        if args.use_wandb:
            import wandb
            wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                       dir=str(out_dir), config=vars(args), resume="allow")

    set_seed(args.seed)
    weight_dtype = {"no": torch.float32, "fp16": torch.float16,
                    "bf16": torch.bfloat16}[args.mixed_precision]

    # ---------------- frozen components ---------------- #
    accel.print("Loading SD3 components ...")
    M = args.pretrained_model
    clip_tok = CLIPTokenizer.from_pretrained(M, subfolder="tokenizer")
    clip_tok2 = CLIPTokenizer.from_pretrained(M, subfolder="tokenizer_2")
    t5_tok = T5TokenizerFast.from_pretrained(M, subfolder="tokenizer_3")
    clip_l = CLIPTextModelWithProjection.from_pretrained(M, subfolder="text_encoder",
                                                         torch_dtype=weight_dtype)
    clip_g = CLIPTextModelWithProjection.from_pretrained(M, subfolder="text_encoder_2",
                                                         torch_dtype=weight_dtype)
    t5 = T5EncoderModel.from_pretrained(M, subfolder="text_encoder_3",
                                        torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(M, subfolder="vae", torch_dtype=weight_dtype)
    transformer = SD3Transformer2DModel.from_pretrained(M, subfolder="transformer",
                                                        torch_dtype=weight_dtype)
    noise_sched = FlowMatchEulerDiscreteScheduler.from_pretrained(M, subfolder="scheduler")
    sched_train = copy.deepcopy(noise_sched)
    sched_eval = copy.deepcopy(noise_sched)

    for m in (clip_l, clip_g, t5, vae, transformer):
        m.requires_grad_(False)
        m.to(accel.device, dtype=weight_dtype).eval()
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    transformer.train()

    # ---------------- LoRA ---------------- #
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, init_lora_weights="gaussian",
        target_modules=LORA_TARGETS[args.lora_targets],
    )
    accel.print(f"LoRA targets: {args.lora_targets}")
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
    lr_sched = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accel.num_processes,
        num_training_steps=args.max_train_steps * accel.num_processes,
    )
    transformer, optimizer, loader, lr_sched = accel.prepare(
        transformer, optimizer, loader, lr_sched)

    vae_scale = vae.config.scaling_factor
    vae_shift = getattr(vae.config, "shift_factor", 0.0) or 0.0

    # eval pipeline over the SAME (LoRA-attached) modules
    pipe = StableDiffusion3Pipeline(
        transformer=accel.unwrap_model(transformer), scheduler=sched_eval, vae=vae,
        text_encoder=clip_l, tokenizer=clip_tok, text_encoder_2=clip_g,
        tokenizer_2=clip_tok2, text_encoder_3=t5, tokenizer_3=t5_tok)
    pipe.set_progress_bar_config(disable=True)

    # ---------------- bench eval setup ---------------- #
    bench, evaluator, fid_ref = None, None, {}
    if args.eval_every > 0:
        bench = load_bench(args.bench_parquet)
        accel.print(f"Bench: {len(bench['texts'])} prompts")
        if accel.is_main_process:
            evaluator = MetricEvaluator(device=str(accel.device))
            fid_ref = load_fid_reference(args.fid_ref_dir)
            if not fid_ref:
                accel.print(f"[warn] no FID reference in {args.fid_ref_dir}")
            with open(out_dir / "bench_prompts.json", "w") as f:
                json.dump(bench, f, indent=2, ensure_ascii=False)

    # ---------------- training loop ---------------- #
    global_step = 0
    if args.resume_from:
        global_step = load_checkpoint(accel, transformer, optimizer, lr_sched,
                                      args.resume_from)
    elif args.init_lora:
        raw = load_file(args.init_lora)
        cleaned = {k.removeprefix("transformer."): v for k, v in raw.items()}
        set_peft_model_state_dict(accel.unwrap_model(transformer), cleaned)
        global_step = int(args.start_step)
        accel.print(f"[init_lora] loaded {args.init_lora}; counting from step {global_step}")
    last_eval_step = -1
    if args.eval_every > 0 and args.eval_on_start and global_step == 0:
        accel.print("[eval] step 0: evaluating base model before training ...")
        accel.wait_for_everyone()
        run_bench_eval(accel, pipe, bench, evaluator, fid_ref, 0, args, out_dir, weight_dtype)
        last_eval_step = 0

    progress = tqdm(total=args.max_train_steps, disable=not accel.is_main_process,
                    initial=global_step, desc="train")
    t_start = time.time()
    running, running_n, epoch = 0.0, 0, 0

    while global_step < args.max_train_steps:
        epoch_completed = True
        for batch in loader:
            with accel.accumulate(transformer):
                pixel_values = batch["pixel_values"].to(accel.device, dtype=weight_dtype,
                                                         non_blocking=True)
                captions = batch["captions"]
                bsz = pixel_values.shape[0]

                prompt_embeds, pooled = encode_prompts_sd3(
                    captions, clip_tok, clip_l, clip_tok2, clip_g, t5_tok, t5,
                    device=accel.device, dtype=weight_dtype, t5_seq_len=args.t5_seq_len)

                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae_shift) * vae_scale

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std,
                    mode_scale=args.mode_scale)
                indices = (u * sched_train.config.num_train_timesteps).long()
                timesteps = sched_train.timesteps[indices].to(accel.device)

                noise = torch.randn_like(latents)
                sigmas = get_sigmas(sched_train, timesteps, n_dim=latents.ndim,
                                    device=accel.device, dtype=latents.dtype)
                noisy = (1.0 - sigmas) * latents + sigmas * noise

                model_pred = transformer(
                    hidden_states=noisy, timestep=timesteps,
                    encoder_hidden_states=prompt_embeds, pooled_projections=pooled,
                    return_dict=False)[0]

                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy
                    target = latents
                else:
                    target = noise - latents
                weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas)
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
                    progress.set_postfix(loss=f"{avg:.4f}", lr=f"{lr_now:.1e}",
                                         steps_per_s=f"{sps:.2f}")
                    if accel.is_main_process and args.use_wandb:
                        import wandb
                        wandb.log({"train/loss": avg, "train/lr": lr_now,
                                   "train/steps_per_sec": sps,
                                   "train/step": global_step}, step=global_step)
                    running, running_n = 0.0, 0

                if global_step % args.save_every == 0:
                    save_checkpoint(accel, transformer, optimizer, lr_sched,
                                    global_step, out_dir)

                if args.eval_every > 0 and global_step % args.eval_every == 0:
                    accel.wait_for_everyone()
                    run_bench_eval(accel, pipe, bench, evaluator, fid_ref,
                                   global_step, args, out_dir, weight_dtype)
                    last_eval_step = global_step

                if global_step >= args.max_train_steps:
                    epoch_completed = False
                    break

        if epoch_completed:
            epoch += 1
            accel.print(f"[epoch] epoch {epoch} finished at step {global_step}")
            save_checkpoint(accel, transformer, optimizer, lr_sched, global_step, out_dir)
            if accel.is_main_process and args.use_wandb:
                import wandb
                wandb.log({"train/epoch": epoch}, step=global_step)
            if args.eval_every > 0 and global_step != last_eval_step:
                accel.wait_for_everyone()
                run_bench_eval(accel, pipe, bench, evaluator, fid_ref,
                               global_step, args, out_dir, weight_dtype)
                last_eval_step = global_step

    if accel.is_main_process:
        _save_lora(accel.unwrap_model(transformer), out_dir, global_step)
        _save_lora(accel.unwrap_model(transformer), out_dir, "final")
        _save_resume_state(optimizer, lr_sched, global_step, out_dir)
        if args.use_wandb:
            import wandb
            wandb.finish()
    accel.print("Training done.")


def _save_lora(transformer, out_dir: Path, tag) -> None:
    """Loadable via StableDiffusion3Pipeline.load_lora_weights."""
    state = get_peft_model_state_dict(transformer)
    prefixed = {f"transformer.{k}": v for k, v in state.items()}
    path = out_dir / f"sd3_lora-{tag}.safetensors"
    save_file(prefixed, str(path))
    print(f"[save] {path} ({sum(v.numel() for v in state.values()):,} params)")


def _save_resume_state(optimizer, lr_sched, step: int, out_dir: Path) -> None:
    path = out_dir / "resume_state.pt"
    torch.save({"step": step, "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_sched.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()}, path)
    print(f"[save] {path} (resume state @ step {step})")


def save_checkpoint(accel, transformer, optimizer, lr_sched, step, out_dir):
    if not accel.is_main_process:
        return
    _save_lora(accel.unwrap_model(transformer), out_dir, step)
    _save_resume_state(optimizer, lr_sched, step, out_dir)


def load_checkpoint(accel, transformer, optimizer, lr_sched, resume_dir: str) -> int:
    rd = Path(resume_dir)
    state_path = rd / "resume_state.pt"
    if not state_path.exists():
        raise FileNotFoundError(f"resume_state.pt not found under {resume_dir}")
    ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
    step = int(ckpt["step"])
    lora_path = rd / f"sd3_lora-{step}.safetensors"
    if not lora_path.exists():
        raise FileNotFoundError(f"matching LoRA weights not found: {lora_path}")
    raw = load_file(str(lora_path))
    cleaned = {k.removeprefix("transformer."): v for k, v in raw.items()}
    set_peft_model_state_dict(accel.unwrap_model(transformer), cleaned)
    optimizer.load_state_dict(ckpt["optimizer"])
    lr_sched.load_state_dict(ckpt["lr_scheduler"])
    try:
        torch.set_rng_state(ckpt["rng"])
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    except Exception as e:
        accel.print(f"[resume] RNG restore skipped: {e}")
    accel.print(f"[resume] step={step} | LoRA={lora_path.name}")
    return step


if __name__ == "__main__":
    main()
