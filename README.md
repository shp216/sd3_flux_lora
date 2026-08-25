# Vector-style T2I LoRA (FLUX.1-dev / SD3.5-medium) on MMSVG, evaluated on MMSVGBench

OmniSVG의 MMSVG 데이터로 FLUX.1-dev(및 SD3.5-medium) LoRA를 학습해 icon / illustration 스타일의
래스터 이미지를 생성하고, OmniSVG와 **동일한 벤치마크·동일한 4개 metric**(FID↓ CLIP↑ Aesthetic↑ HPS↑)으로
비교하는 프로젝트입니다. 이 문서는 **다른 서버에서 (Claude가) 동일한 환경·데이터·평가·학습을 그대로 재현**할 수
있도록 쓰였습니다. 데이터/가중치/체크포인트는 git에 포함하지 않으며(.gitignore), 아래 절차로 다시 만듭니다.

---

## 0. 다른 서버용 체크리스트 (TL;DR)

아래 순서대로 실행하면 이 서버와 같은 상태가 됩니다. 각 단계의 세부는 해당 절 참고.

1. 저장소 clone → `ROOT`로 둡니다 (여기서는 `/data/shp216/Flux-Lora-train-bundle`).
2. **Docker 이미지 빌드/컨테이너 실행** (§2) — 호스트 uid/gid로 실행, `$ROOT`와 HF 캐시 경로 마운트.
3. **HF 토큰 로그인** (§2.3): gated repo 접근 필요 — `black-forest-labs/FLUX.1-dev`, `OmniSVG/MMSVG-Icon`,
   `stabilityai/stable-diffusion-3.5-medium`(SD3.5 학습 시). `wandb login`도.
4. **데이터 다운로드** (§3.1): MMSVG-Icon(18GB) + MMSVG-Illustration(12GB) + MMSVGBench.
5. **학습 데이터 추출** (§3.2): `extract_mmsvg.py` → 30만 장(icon 15만 + illu 15만), 1024² 흰 배경 렌더,
   캡션 = `<vector> ` + description. 결과 `train_data/manifest.jsonl` + `train_data/images/`.
6. **FID 레퍼런스 생성** (§3.3): `build_fid_ref.py --bg white` → `train_data/fid_ref_white/{icon,illustration,all}.npz`.
   (`--bg white`가 핵심 — §3.3의 이유 참고.)
7. (권장) **metric 파이프라인 검증** (§3.4): `tools/_smoke_metrics.py` 실행, 4개 metric이 모두 계산되는지 확인.
8. **스모크 학습** (§5.3) 12스텝 + eval 1회 → 통과 후 **본 학습** `train/run_flux.sh` 또는 `train/run_sd3.sh`.
9. 비교 기준선(§7)은 코드에 상수로 박혀 있어 wandb에 자동으로 가로선으로 찍힙니다.

---

## 1. 디렉토리 구성

```
ROOT/
├── README.md
├── .gitignore
├── docker/Dockerfile              # 학습·평가 환경 (CUDA 12.8 / torch 2.9.1 + 모든 의존성)
├── train/
│   ├── train_flux_lora.py         # FLUX.1-dev LoRA 학습 + 벤치 eval + wandb + resume
│   ├── train_sd3_lora.py          # SD3.5-medium LoRA 학습 (동일 데이터/eval 프로토콜)
│   ├── run_flux.sh / run_sd3.sh   # 최종 레시피 실행 스크립트 (env 변수로 오버라이드)
│   ├── dataset.py                 # manifest.jsonl 로더 (Resize→CenterCrop→[-1,1])
│   ├── extract_mmsvg.py           # parquet → SVG 1024 렌더 + manifest 생성
│   ├── build_fid_ref.py           # FID 레퍼런스 통계(mu/sigma) 사전 계산
│   ├── bench_metrics.py           # FID / CLIP / Aesthetic / HPS (OmniSVG 구현과 수치 동일)
│   ├── score_omnisvg.py           # OmniSVG 생성물 채점 (논문 재현용)
│   ├── select_best_clip.py        # (재현 실험용) 후보 중 CLIP 최고 1장 선택
│   ├── requirements.txt
│   └── tools/                     # 진단 스크립트 (경로가 하드코딩되어 있음, 참고용)
├── train_data/   (git 제외)  raw/ images/ manifest.jsonl fid_ref_white/
├── ckpt/         (git 제외)  run-*/ (체크포인트, eval 결과, args.json), *.log
└── omnisvg_repro/(git 제외)  OmniSVG 8B 생성물 및 채점 결과
```

---

## 2. 환경 (Docker)

### 2.1 하드웨어/소프트웨어
- 이 서버: NVIDIA RTX PRO 6000 Blackwell 96GB × 8, driver CUDA 12.8. Blackwell(sm_120)은 **torch ≥ 2.7 cu128** 필요.
- 베이스 이미지 `pytorch-claude:2.9.1-cu128`(torch 2.9.1+cu128, torchvision 0.24.1, python 3.11)는 이 서버 로컬 이미지입니다.
  다른 서버에서는 `docker/Dockerfile`의 `FROM`을 **동급 공식 이미지**로 바꾸세요, 예:
  `FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel` (torch/torchvision 포함). 나머지 레이어는 그대로 동작합니다.
- 검증된 패키지 버전: diffusers 0.40.0, transformers 5.15.1, accelerate 1.14.0, peft 0.20.0, datasets 5.0.1, wandb 0.28.

### 2.2 빌드 & 실행
```bash
# Dockerfile 안의 uid/gid(1016)는 이 서버 사용자 기준 → 다른 서버에서는 `id -u`/`id -g` 값으로 바꿀 것
docker build -t flux-lora:cu128 docker/
docker run -d --name flux-lora --gpus all --ipc=host \
  -v /data/shp216:/data/shp216 \          # ROOT와 hf_cache가 들어있는 경로
  flux-lora:cu128 sleep infinity
docker exec -it flux-lora bash            # 접속
```
- `--ipc=host` 필수 (DataLoader 워커 공유메모리). 컨테이너 사용자를 호스트 uid와 맞추지 않으면 마운트 경로에 쓰기 실패.
- `HF_HOME=/data/shp216/hf_cache`, `TORCH_HOME=/data/shp216/hf_cache/torch`는 Dockerfile에 ENV로 설정 → 모델/데이터 캐시가 호스트에 남음.
  다른 서버에서 ROOT가 다르면 Dockerfile ENV와 아래 명령의 경로를 함께 바꾸세요.
- 학습은 **호스트 tmux**에서 `docker exec ...`를 실행하는 방식 (컨테이너에 tmux 없음): 예 `tmux new -d -s flux "docker exec flux-lora bash -c '...'"`.

### 2.3 인증
```bash
docker exec -it flux-lora hf auth login      # 토큰은 HF_HOME/token 에 저장 → 컨테이너 재생성에도 유지
docker exec -it flux-lora wandb login        # ~/.netrc (컨테이너 홈) — 컨테이너 재생성 시 다시 필요
```
gated repo: FLUX.1-dev, MMSVG-Icon, SD3.5-medium 은 HF 웹에서 라이선스 동의가 먼저 필요합니다.

### 2.4 metric 의존성의 함정 (Dockerfile에 반영됨)
- `openai-clip`, `hpsv2`는 **`pip install --no-deps`** (일반 설치 시 torch를 다른 빌드로 덮어씀).
- hpsv2 sdist에 BPE vocab이 빠져 있음 → `clip/bpe_simple_vocab_16e6.txt.gz`를 `hpsv2/src/open_clip/`로 복사.
- cairosvg는 시스템 `libcairo2` 필요.
- transformers ≥ 5: `CLIPModel.get_image_features()`가 텐서가 아니라 출력 객체를 반환 → `bench_metrics.py`가 `.pooler_output`으로 처리(버전 양쪽 호환).

---

## 3. 데이터

### 3.1 다운로드 (컨테이너 안에서)
```bash
R=$ROOT/train_data/raw
hf download OmniSVG/MMSVG-Illustration --repo-type dataset --local-dir $R/MMSVG-Illustration   # 11.8GB, parquet 26개
hf download OmniSVG/MMSVG-Icon         --repo-type dataset --local-dir $R/MMSVG-Icon           # 18.7GB, parquet 91개 (gated)
hf download OmniSVG/MMSVGBench         --repo-type dataset --local-dir $R/MMSVGBench           # text2svg 300 + image2svg
hf download black-forest-labs/FLUX.1-dev --exclude 'flux1-dev.safetensors' 'ae.safetensors'      # diffusers 포맷 ~32GB
hf download stabilityai/stable-diffusion-3.5-medium --exclude 'sd3.5_medium.safetensors' 'text_encoders/*'  # SD3.5용
```
parquet 스키마 (Icon/Illustration 공통): `id, svg, description(짧은 캡션), keywords, detail(긴 캡션), image(896² RGBA PNG 렌더), token_len`.
전체 행 수: Icon 904,011 / Illustration 255,412. 벤치 text2svg: 300개 (icon 150 + illustration 150), 프롬프트 3~13단어(중앙값 6).

### 3.2 학습 데이터 추출 (최종 설정)
```bash
cd $ROOT/train
python extract_mmsvg.py --raw_root $ROOT/train_data/raw --out_dir $ROOT/train_data \
    --n_icon 150000 --n_illu 150000 --resolution 1024 --workers 32 --trigger "<vector>" --seed 42
```
- **샘플링**: 각 데이터셋에서 샤드 행수 비례로 seed 42 랜덤 15만 장씩 → 총 30만 장 (1:1 — 벤치의 150:150 비율과 정합).
- **해상도**: parquet의 896² PNG를 쓰지 않고 **SVG를 cairosvg로 1024²에 직접 렌더링**(벡터라 업스케일 손실 없음), 흰 배경 합성(`background_color="white"`), 렌더 실패 0.
- **캡션**: `description` 필드(중앙값 13~14단어; 벤치 프롬프트 스타일과 가장 유사) 앞에 트리거 **`<vector> `**를 붙임.
  `detail`(긴 캡션, ~55단어)도 manifest에 보관되어 있으나 학습에는 미사용.
- manifest 한 줄 예:
  `{"id": "...", "image": "images/icon/00001/<id>.png", "caption": "<vector> An illustration of a black-outlined battery icon ...", "detail": "...", "src": "icon"}`
  (image 경로는 manifest 디렉토리 기준 상대경로 → `dataset.py`가 해석, 서버 간 이동 가능)
- 디스크: 이미지 11GB. 32 프로세스로 약 10~15분.
- **정확히 같은 30만 장을 재현하려면** (다른 서버에서 SD3.5 등을 같은 데이터로 학습할 때): seed 재현 대신 저장소에 포함된
  `data/dataset_manifest_compact.json.gz`(각 항목의 원본 parquet 파일 + 행 번호 30만 개, ~10MB)로 렌더하세요.
  ```bash
  python extract_mmsvg.py --raw_root $ROOT/train_data/raw --out_dir $ROOT/train_data \
      --from_manifest $ROOT/data/dataset_manifest_compact.json.gz --resolution 1024 --workers 32 --trigger "<vector>"
  ```
  같은 parquet(동일 HF 리비전)만 있으면 파일명(id)·캡션·이미지가 이 서버와 동일하게 생성됩니다. 검증됨: 재계획 결과 300,000개, 표본 샤드의 id 전수 일치.
  이 JSON에는 FID 레퍼런스의 행 인덱스(`fid_reference.*.row_indices_in_load_dataset_order`)도 들어 있습니다.

### 3.3 FID 레퍼런스 통계 (중요)
```bash
python build_fid_ref.py --raw_root $ROOT/train_data/raw --out_dir $ROOT/train_data/fid_ref_white --bg white
```
- OmniSVG 이슈 프로토콜 그대로: 각 데이터셋 전체를 `load_dataset` 행 순서로 놓고 `np.random.seed(42)`,
  `np.random.choice(N, int(N*0.03), replace=False)` → **Icon 27,120장 / Illustration 7,662장**, 데이터셋의 `image` 컬럼(896² 렌더) 사용,
  torchvision InceptionV3(IMAGENET1K_V1, fc 제거, 299² 리사이즈) 특징의 mu/sigma 저장. `all.npz`는 두 세트 concat.
- **`--bg white`인 이유**: GT PNG가 투명 배경 RGBA인데, 공개 `compute_fid.py`의 `convert('RGB')`는 알파를 버려 투명 영역이
  **검정**이 됩니다. 그 방식(`--bg raw`)으로는 OmniSVG 4B/8B 생성물의 FID가 200±(논문 137/154)로 나오고 icon/illu 순서까지 뒤집힙니다.
  흰 배경 합성(`--bg white`)으로 재면 논문 스케일·순서가 복원됩니다(§7). 본 프로젝트의 FID는 **모두 흰 배경 레퍼런스** 기준입니다.
- 검증 완료 사항: 이 구현은 원본 `compute_fid.py`와 소수점 4자리까지 동일(213.7400 vs 213.73 등), 라이브러리 버전(torch 2.3 vs 2.9)은 결과에 무영향.
  n=150 소표본 FID의 바닥값(진짜 GT 150장): icon 93.1 / illu 117.9 — 이 아래로는 내려갈 수 없음.

### 3.4 metric 파이프라인 검증 (권장)
```bash
cd $ROOT/train && PYTHONPATH=. python tools/_smoke_metrics.py   # 학습 렌더 12장으로 4개 metric 전부 계산
```
모델 가중치가 이때 캐시됩니다: `openai/clip-vit-base-patch32`, OpenAI CLIP ViT-L/14(+LAION aesthetic MLP `sac+logos+ava1-l14-linearMSE.pth`),
HPSv2 `xswu/HPSv2/HPS_v2_compressed.pt`, torchvision InceptionV3.

---

## 4. 평가 프로토콜 (정확한 정의)

학습 스크립트 안에서 자동 실행되며, OmniSVG 비교와 조건을 맞춘 것입니다.

| 항목 | 값 |
|---|---|
| 프롬프트 | MMSVGBench text2svg 300개 원문 (icon 150 + illustration 150) |
| 생성 | 프롬프트당 **1장**, seed = 프롬프트 인덱스(스텝 간 비교 가능), 1024², 28 스텝, guidance FLUX 3.5 / SD3.5 7.0 |
| 트리거 | **생성 시에만** `--eval_trigger "<vector>"`로 앞에 붙임. **CLIP/HPS 채점은 벤치 원문(트리거 없음)** |
| FID | 흰 배경 레퍼런스(§3.3) 대비, icon 150장 vs icon ref / illu 150장 vs illu ref, 수식은 OmniSVG와 동일(eps 처리 포함) |
| CLIP | `openai/clip-vit-base-patch32` 이미지-텍스트 코사인 유사도 (OmniSVG `compute_clip.py`와 동일) |
| Aesthetic | LAION improved-aesthetic-predictor (OpenAI CLIP ViT-L/14 fp16 임베딩 + MLP) |
| HPS | HPSv2 v2.0 (ViT-H-14, 체크포인트 가중치로 전체 교체) |
| 리포트 값 | `*_mean = (icon + illustration) / 2` — 논문 표에는 이 값을 씀 |
| 빈도 | step 0(학습 전 base) + `--eval_every 500` + 매 epoch 종료 시; 체크포인트도 같은 시점에 저장 |
| 산출물 | `ckpt/<run>/eval/step{N}/gen/000..299.png`, `metrics.json`, `grid-0..2.jpg`(10×10, 100장씩) |
| wandb | `eval/{fid,clip,aesthetic,hps}_{icon,illustration,mean}`, `eval/fid_all`, `eval/grid_{0,1,2}`, 그리고 **`eval/*_omnisvg8b`**(OmniSVG 8B 상수 시리즈 → 같은 패널에 겹쳐 가로선으로 비교) |

eval 비용: 8 GPU 기준 300장 생성 ~6분 + metric ~1분.

---

## 5. 학습 — FLUX.1-dev (최종 레시피)

| 항목 | 값 | 비고 |
|---|---|---|
| 베이스 | `black-forest-labs/FLUX.1-dev` bf16, 전부 frozen | LoRA 파라미터만 fp32로 학습 |
| LoRA 타깃 | **`--lora_targets all-linear`(기본)**: kohya sd-scripts / ostris ai-toolkit 기본과 동일 — double 19 + single 38 블록 안의 **모든 Linear** (attention q/k/v/out + context, FFN, single-block `proj_mlp`/`proj_out`, AdaLN modulation `norm*.linear`), 494층. 정규식 full-match라 최상위 `proj_out`/임베더/`norm_out`은 제외. `--lora_targets official` = diffusers 예제 기본(double attention+FFN, single attention만) | rank 128: all-linear 687.3M / official 358.6M params |
| rank / alpha | 128 / 128 (rank 64 대비 우세 확인) | `init_lora_weights="gaussian"` |
| 해상도 | 1024 | Resize(1024)→CenterCrop |
| effective batch | **64** = per-GPU 8 × 8 GPU × accum 1 | 4 GPU면 per-GPU 16 (실측 76GB/96GB @ rank128) |
| LR | 1e-4, AdamW(wd 1e-4), `constant_with_warmup` warmup 200, grad clip 1.0 | warmup 중 lr가 1e-5 등으로 보이는 건 정상 |
| timestep | flow matching, `logit_normal(0,1)` 밀도 샘플링, loss weighting 1 | SD3 논문 레시피 (diffusers FLUX 예제 기본값은 `none`) |
| guidance(학습) | 1.0 (guidance-distilled 모델 파인튜닝 관례) | eval은 3.5 |
| T5 길이 | 128 | |
| 스텝 | 15,000 (= 3.2 epoch; 1 epoch = 300,000/64 = 4,687 스텝) | |
| 속도/메모리 | 8 GPU × batch 8 @1024: ~9.7 s/step, 58GB/GPU (grad checkpointing on) | 1 epoch ≈ 12.6h |

### 5.1 실행
```bash
cd $ROOT/train
tmux new -d -s flux "docker exec flux-lora bash -c 'cd $ROOT/train && ./run_flux.sh'"
# 오버라이드 예: RANK=64 RUN=run-flux-rank64 ./run_flux.sh / NPROC=4 GPUS=0,1,2,3 BATCH=16 ./run_flux.sh
# resume:  RESUME=$ROOT/ckpt/run-flux-rank128 ./run_flux.sh   (resume_state.pt의 step + 해당 LoRA에서 이어감)
```
`run_flux.sh`는 `--bench_parquet`/`--fid_ref_dir`를 `$ROOT` 기준으로 넘깁니다 (스크립트 기본값은 이 서버 경로로 하드코딩되어 있으니 항상 넘길 것).

### 5.2 체크포인트
- `flux_lora-{step}.safetensors`: `transformer.` 접두 키 → `FluxPipeline.load_lora_weights(dir, weight_name=...)`로 바로 로드 가능.
- `resume_state.pt`: optimizer + LR 스케줄러 + step (매 저장 시 덮어씀). 데이터 순서는 복원되지 않음(알려진 한계).
- resume 시 epoch 카운터는 resume 시점부터 셉니다(epoch-end eval 시점이 이동).

### 5.3 스모크 테스트 (새 서버에서 본 학습 전에)
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes 1 --mixed_precision bf16 train_flux_lora.py \
  --manifest $ROOT/train_data/manifest.jsonl --output_dir $ROOT/ckpt/smoke \
  --bench_parquet $ROOT/train_data/raw/MMSVGBench/data/text2svg-00000-of-00001.parquet \
  --fid_ref_dir $ROOT/train_data/fid_ref_white \
  --resolution 1024 --batch_size 4 --max_train_steps 12 --save_every 10 --log_every 2 \
  --eval_every 10 --eval_trigger "<vector>" --eval_resolution 512 --eval_inference_steps 4
```
학습→저장→300장 생성→4 metric→그리드까지 한 번에 검증(약 20분). 통과하면 `ckpt/smoke/eval/step0000010/`에 산출물이 생깁니다.

---

## 6. 학습 — SD3.5-medium (`train_sd3_lora.py`, `run_sd3.sh`)

FLUX 스크립트와 데이터·eval·wandb·체크포인트 로직이 동일하고, 모델 고유 부분만 diffusers 공식
`train_dreambooth_lora_sd3.py`를 따릅니다:

| 항목 | FLUX | SD3.5-medium |
|---|---|---|
| 텍스트 인코딩 | CLIP-L pooled + T5 시퀀스 | CLIP-L + CLIP-G (penultimate hidden, concat→4096 zero-pad) ++ T5 시퀀스; pooled = concat(CLIP-L, CLIP-G projection) |
| latent | VAE 16ch → 2×2 패킹 (B, S, 64) | VAE 16ch (B,16,H/8,W/8), 패킹 없음, `(z - shift_factor) * scaling_factor` |
| timestep 입력 | t/1000 | t (0..1000) |
| 학습 목표 | v = noise − x0 | `--precondition_outputs 1`(공식 기본): pred·(−σ)+noisy vs x0 |
| LoRA 타깃 | `all-linear`(기본): 블록 내 모든 Linear, 494층 / `official`: diffusers 예제 기본 | `all-linear`(기본): 블록 내 모든 Linear — attn + **attn2(MMDiT-X dual attention)** + FFN + modulation, 385층 (rank128 260.9M) / `official`: attention만, 191층 (75.1M) |
| eval 생성 | 수동 Euler 루프, guidance 3.5 (distilled) | `StableDiffusion3Pipeline` 호출, **true CFG guidance 7.0** (파이프라인 기본) |
| T5 길이 | 128 | 77 (공식 기본) |
| 체크포인트 | `flux_lora-{step}.safetensors` | `sd3_lora-{step}.safetensors` (`StableDiffusion3Pipeline.load_lora_weights` 호환) |

```bash
tmux new -d -s sd3 "docker exec flux-lora bash -c 'cd $ROOT/train && ./run_sd3.sh'"
```
- 기본 rank 128 / effective batch 64 / LR 1e-4 / 15,000 스텝 — FLUX와 동일 조건으로 맞춰 두었고, SD3.5-medium(2.5B)은 훨씬 가벼워 per-GPU batch를 올릴 여지가 큽니다.
- **주의**: 이 스크립트는 작성 후 문법 검사 및 (가능했다면) 소규모 스모크만 거쳤습니다. 새 서버에서 §5.3과 같은 12스텝 스모크(`train_sd3_lora.py`로 교체)를 먼저 돌려 eval까지 통과하는지 확인하세요.

---

## 7. 기준선과 지금까지의 관찰

### 7.1 OmniSVG 8B 기준선 (코드 상수 `OMNISVG8B_BASELINE`)
`OmniSVG/OmniSVG1.1_8B`(Qwen2.5-VL-7B 베이스)를 공식 `inference.py`(기본 샘플링 파라미터)로 벤치 300개에 돌리고
**첫 번째 유효 후보 1장**(299장; 1개 프롬프트는 유효 SVG 실패)을 512² 흰 배경으로 렌더해 위 프로토콜로 채점:

| | FID↓ | CLIP↑ | Aesthetic↑ | HPS↑ |
|---|---|---|---|---|
| icon | 119.06 | 0.274 | 4.596 | 0.244 |
| illustration | 159.66 | 0.213 | 4.523 | 0.221 |
| **mean** | **139.36** | **0.2437** | **4.560** | **0.2329** |

논문(3B, text2svg) 값은 icon 137.40/0.275/4.62/0.244, illustration 154.37/0.226/4.56/0.232 — CLIP/Aesthetic/HPS는 재현되고
FID는 흰 배경 프로토콜에서 같은 스케일·순서로 재현됩니다(검정 배경 프로토콜에서는 213/178로 불일치).
재현 절차: `omnisvg:cu128` 이미지(transformers 4.51.3, qwen_vl_utils)에서 `/data/shp216/OmniSVG` 레포의 `inference.py --task text-to-svg --model-size 8B --save-png`,
청크별 병렬 실행 후 `score_omnisvg.py <fid_ref_dir> <gen_dir>`.

### 7.1b LoRA 적용 범위 조사 (prior work)
| 구현 | 기본 LoRA 대상 |
|---|---|
| kohya sd-scripts `networks/lora_flux.py` | DoubleStreamBlock + SingleStreamBlock 내 모든 Linear (attn, img/txt MLP, single linear1/linear2, modulation) |
| ostris ai-toolkit (FLUX 예제 config) | 제한 없음 → 블록 내 모든 Linear |
| SimpleTuner `--flux_lora_target` | `all`=attention, `all+ffs`=attention+FFN(“objective 적응에 도움”), `mmdit`=안정적이나 느림 |
| diffusers `train_dreambooth_lora_flux.py` | double attention+FFN, single attention q/k/v만 |
→ 본 프로젝트 기본값은 kohya/ai-toolkit과 같은 **all-linear**. (official 타깃으로 돌린 트리거 런은 step 1,500에서 백지/단색 붕괴가 관측됨 — §7.2)

### 7.2 FLUX 학습 관찰 (rank 128, effective 64, LR 1e-4)
- step 0 (base FLUX): fid_mean 239.0 / clip 0.322 / aesthetic 5.74 / hps 0.290 — CLIP·Aesthetic·HPS는 base부터 OmniSVG를 상회, **FID만 크게 뒤짐**(벡터 도메인 분포와의 거리).
- 트리거 없이 학습: fid_mean 204 (1k) → 194 (2k) → **184.6 (4.5k, best)** 이후 190±8에서 정체(7k까지). rank 64는 3k 이후 상승(210).
- 정체 원인(측정): 생성물의 흰 배경 비율 icon 49% / illu 28% (학습 데이터 92% / 88%), 어두운 배경 23~30%, 채도 2~3배.
  학습 캡션의 33%가 "white background"를 명시하고 벤치 프롬프트는 0% → 스타일이 **캡션 조건부**로 학습되어 짧은 벤치 프롬프트에서 발현되지 않음.
- 대응: 트리거 `<vector>` 복원(학습 캡션 + eval 생성 프롬프트). step 0에서 트리거만으로 base FLUX FID 239.0 → 224.2.
  (OmniSVG도 벤치 텍스트를 시스템 프롬프트+지시문으로 감싸 생성하므로 고정 접두 토큰은 비교상 공정.)

---

## 8. 알려진 함정 / 운영 메모
- **NCCL 타임아웃**: main 프로세스만 metric을 계산하는 동안 다른 rank가 대기 → `InitProcessGroupKwargs(timeout=3h)` + eval 전후 `wait_for_everyone()`로 처리됨.
- **kill -9 후 GPU util 100% 잔상**: 메모리 0·프로세스 없음·전력 유휴면 드라이버 표시 잔상. 각 GPU에 작은 CUDA 작업을 한 번 돌리면 0%로 리셋.
- **백지 이미지**: 저내용 프롬프트가 ~2% 백지로 수렴하는 경향. `tools/` 스크립트로 mean>250 비율 집계 가능.
- **FID 노이즈**: n=150 프로토콜에서 스텝 간 ±7~10 진동은 정상. 단일 반등에 반응하지 말고 3회 연속 추세로 판단.
- **로그**: 학습 로그는 `ckpt/<run>.log`(tqdm `\r` 포함 → `tr '\r' '\n'`로 읽기), eval 결과는 `[eval] step N: fid_mean=...` 라인.
- **wandb 기준선 겹쳐 보기**: 패널에서 `eval/fid_mean`과 `eval/fid_mean_omnisvg8b`를 같이 선택.

## 9. 재현 산식 요약
- 1 epoch = 300,000 / effective batch. effective 64 → 4,687 스텝.
- 흰 배경 FID 레퍼런스 = Icon 3%(27,120) / Illustration 3%(7,662), seed 42, InceptionV3 pool 2048-d.
- 벤치 생성 seed = 프롬프트 인덱스(0..299) → 스텝/모델 간 그리드가 같은 seed로 비교됨.
