"""Version-sensitivity check for FID: run the SAME images through OmniSVG's
own InceptionV3Feature class (imported unmodified from their metrics dir) and
print features + FID. Executed once in the pinned env (torch 2.3.0 / tv 0.18 /
numpy 2.2.6 / Pillow 10.1, CPU) and once in our env (torch 2.9.1, CPU);
identical outputs kill the version hypothesis.
"""
import io
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision
from PIL import Image

sys.path.insert(0, "/data/shp216/OmniSVG/metrics")
from compute_fid import (InceptionV3Feature, calculate_activation_statistics,
                         calculate_frechet_distance)

print(f"env: torch={torch.__version__} torchvision={torchvision.__version__} "
      f"numpy={np.__version__} pillow={Image.__version__ if hasattr(Image, '__version__') else 'n/a'}")

# fixed inputs: first 200 GT icon images (as-is, their convert path) + 150 gens
shard = ("/data/shp216/Flux-Lora-train-bundle/train_data/raw/"
         "MMSVG-Icon/data/train-00000-of-00091.parquet")
t = pq.read_table(shard, columns=["image"]).slice(0, 200)
gt_imgs = [Image.open(io.BytesIO(t.column("image")[i].as_py()["bytes"]))
           for i in range(200)]

import glob
gen_paths = sorted(glob.glob(
    "/data/shp216/Flux-Lora-train-bundle/omnisvg_repro/gen_icon/*.png"))
print(f"inputs: {len(gt_imgs)} GT, {len(gen_paths)} generated")

fe = InceptionV3Feature(device="cpu")
torch.manual_seed(0)

gt_feats = fe.extract_features_from_pil_images(gt_imgs, batch_size=50)
gen_feats = fe.extract_features_batch(gen_paths, batch_size=50)

mu1, s1 = calculate_activation_statistics(gt_feats)
mu2, s2 = calculate_activation_statistics(gen_feats)
fid = calculate_frechet_distance(mu1, s1, mu2, s2)

print(f"gt_feats  sum={gt_feats.sum():.6f} mean={gt_feats.mean():.8f}")
print(f"gen_feats sum={gen_feats.sum():.6f} mean={gen_feats.mean():.8f}")
print(f"FID(gen150 vs gt200) = {fid:.6f}")
print("ENV_FID_CHECK_OK")
