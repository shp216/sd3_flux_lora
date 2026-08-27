#!/usr/bin/env bash
# SD3.5-medium LoRA -- same data + eval protocol as the FLUX run.
#   ./run_sd3.sh                      # fresh run, 8 GPUs
#   NPROC=4 GPUS=0,1,2,3 BATCH=16 ./run_sd3.sh   # 4 GPUs (effective batch stays 64)
#   RESUME=/path/to/ckpt/run-x ./run_sd3.sh       # resume (LoRA + optimizer + sched)
# effective batch = BATCH x NPROC x ACCUM  (default 8 x 8 x 1 = 64)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="${ROOT:-$(cd .. && pwd)}"
NPROC="${NPROC:-8}"; GPUS="${GPUS:-0,1,2,3,4,5,6,7}"; PORT="${PORT:-29601}"
BATCH="${BATCH:-8}"; ACCUM="${ACCUM:-1}"; RANK="${RANK:-128}"; LR="${LR:-1e-4}"
STEPS="${STEPS:-15000}"; TRIGGER="${TRIGGER:-<vector>}"; TARGETS="${TARGETS:-attn-ffn}"
MANIFEST="${MANIFEST:-$ROOT/train_data/manifest.jsonl}"
EVAL_GUIDANCE="${EVAL_GUIDANCE:-4.5}"   # SD3.5-medium model-card default (diffusers pipeline default is 7.0)
RUN="${RUN:-run-sd3-rank$RANK}"; OUT="$ROOT/ckpt/$RUN"
RESUME_ARGS=""; [[ -n "${RESUME:-}" ]] && RESUME_ARGS="--resume_from $RESUME"

CUDA_VISIBLE_DEVICES=$GPUS PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
accelerate launch --num_processes $NPROC --mixed_precision bf16 --main_process_port $PORT \
  train_sd3_lora.py \
    --pretrained_model stabilityai/stable-diffusion-3.5-medium \
    --manifest      $MANIFEST \
    --output_dir    $OUT \
    --bench_parquet $ROOT/train_data/raw/MMSVGBench/data/text2svg-00000-of-00001.parquet \
    --fid_ref_dir   $ROOT/train_data/fid_ref_white \
    --resolution 1024 --batch_size $BATCH --gradient_accumulation_steps $ACCUM --num_workers 8 \
    --learning_rate $LR --max_train_steps $STEPS \
    --lora_rank $RANK --lora_alpha $RANK --lora_targets $TARGETS --t5_seq_len 77 --mixed_precision bf16 \
    --save_every 500 --log_every 20 \
    --eval_every 500 --eval_on_start --eval_trigger "$TRIGGER" \
    --eval_resolution 1024 --eval_inference_steps 28 --eval_guidance_scale $EVAL_GUIDANCE \
    --use_wandb --wandb_project flux-lora-omnisvg --wandb_run_name "$RUN" \
    $RESUME_ARGS 2>&1 | tee -a "$ROOT/ckpt/$RUN.log"
