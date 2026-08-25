FROM pytorch-claude:2.9.1-cu128

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/data/shp216/hf_cache \
    TORCH_HOME=/data/shp216/hf_cache/torch \
    PIP_NO_CACHE_DIR=1

USER root

RUN apt-get update -q && apt-get install -y -q libcairo2 && rm -rf /var/lib/apt/lists/*

RUN pip install \
    "diffusers>=0.38.0" \
    "transformers>=4.49.0" \
    "accelerate>=1.4.0" \
    "peft>=0.19.0" \
    "bitsandbytes>=0.43.3" \
    "sentencepiece>=0.2.0" \
    "protobuf>=4.25" \
    "safetensors>=0.4.5" \
    "einops>=0.8.0" \
    "huggingface_hub[hf_transfer]" \
    "datasets>=3.0.0" \
    pyarrow \
    pillow \
    wandb \
    tqdm \
    cairosvg \
    scipy \
    ftfy \
    regex \
    clint \
    "timm>=0.9"

# metric deps: install without deps so pip can't touch the pinned torch build,
# and patch the BPE vocab file missing from the hpsv2 sdist
RUN pip install --no-deps openai-clip hpsv2 && \
    cp /opt/conda/lib/python3.11/site-packages/clip/bpe_simple_vocab_16e6.txt.gz \
       /opt/conda/lib/python3.11/site-packages/hpsv2/src/open_clip/

# match host user shp216 (uid/gid 1016) so files written to the /data mount stay owned by the host user
RUN groupadd -g 1016 shp216 && useradd -m -u 1016 -g 1016 -s /bin/bash shp216
USER shp216
ENV HOME=/home/shp216

WORKDIR /data/shp216
