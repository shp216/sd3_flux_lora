"""MMSVGBench metrics, faithful to OmniSVG's reference implementations
(/data/shp216/OmniSVG/metrics): FID (torchvision InceptionV3 + scipy sqrtm),
CLIP score (openai/clip-vit-base-patch32), Aesthetic (LAION improved-aesthetic
-predictor on CLIP ViT-L/14), HPS (HPSv2, ViT-H-14).

Differences from the reference scripts are operational only:
  * models load once and batch over images (theirs reload per image),
  * models can be offloaded to CPU between evals to free training VRAM,
  * HPSv2's ViT-H is built randomly-initialized instead of downloading the
    4GB laion2B pretrain, since the HPSv2 checkpoint replaces every weight.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torchvision import models, transforms

CACHE_ROOT = Path(os.environ.get("HF_HOME", "/data/shp216/hf_cache"))
AESTHETIC_URL = ("https://github.com/christophschuhmann/improved-aesthetic-predictor/"
                 "raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth")


# --------------------------------------------------------------- FID math #
def activation_statistics(features: np.ndarray):
    return np.mean(features, axis=0), np.cov(features, rowvar=False)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Identical to OmniSVG compute_fid.calculate_frechet_distance."""
    diff = mu1 - mu2
    dot_product = np.sum(diff * diff)
    sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
    sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except Exception:
        A = sigma1.dot(sigma2)
        eigenvalues, eigenvectors = linalg.eigh(A)
        eigenvalues = np.maximum(eigenvalues, 0)
        covmean = eigenvectors.dot(np.diag(np.sqrt(eigenvalues))).dot(eigenvectors.T)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return dot_product + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


# ----------------------------------------------------- aesthetic MLP head #
class AestheticMLP(nn.Module):
    """Same architecture/state-dict layout as the reference (minus lightning)."""

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _normalized(a: np.ndarray, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def _load_images(paths: list[str]) -> list[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]


class MetricEvaluator:
    """Loads all four metric models lazily; .to_gpu()/.offload() move them
    between the eval device and CPU so training VRAM is only borrowed
    while metrics run."""

    def __init__(self, device: str = "cuda", batch_size: int = 50):
        self.device = device
        self.batch_size = batch_size
        self._loaded = False

    # ------------------------------------------------------------ loading #
    def _ensure_loaded(self):
        if self._loaded:
            return
        import clip as openai_clip
        from transformers import CLIPModel, CLIPProcessor

        # 1) FID feature extractor
        self.inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.inception.fc = nn.Identity()
        self.inception.eval()
        self.fid_tf = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # 2) CLIP score (ViT-B/32, HF transformers)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()

        # 3) Aesthetic: OpenAI CLIP ViT-L/14 + LAION MLP head
        #    load on CPU fp32, run on GPU fp16 (matches clip.load(device="cuda"))
        self.ae_clip, self.ae_preprocess = openai_clip.load("ViT-L/14", device="cpu")
        self.ae_clip.eval()
        ae_path = CACHE_ROOT / "aesthetic_predictor" / "sac+logos+ava1-l14-linearMSE.pth"
        if not ae_path.exists():
            ae_path.parent.mkdir(parents=True, exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(AESTHETIC_URL, ae_path)
        self.ae_mlp = AestheticMLP(768)
        self.ae_mlp.load_state_dict(torch.load(ae_path, map_location="cpu",
                                               weights_only=True))
        self.ae_mlp.eval()

        # 4) HPSv2 (ViT-H-14; weights come entirely from the HPS checkpoint)
        import huggingface_hub
        from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
        self.hps_model, _, self.hps_preprocess = create_model_and_transforms(
            "ViT-H-14", "", precision="amp", device="cpu", jit=False,
            force_quick_gelu=False, force_custom_text=False,
            force_patch_dropout=False, force_image_size=None,
            pretrained_image=False, image_mean=None, image_std=None,
            light_augmentation=True, aug_cfg={}, output_dict=True,
            with_score_predictor=False, with_region_predictor=False,
        )
        cp = huggingface_hub.hf_hub_download("xswu/HPSv2", "HPS_v2_compressed.pt")
        ckpt = torch.load(cp, map_location="cpu", weights_only=False)
        self.hps_model.load_state_dict(ckpt["state_dict"])
        self.hps_tokenizer = get_tokenizer("ViT-H-14")
        self.hps_model.eval()

        self._loaded = True

    def to_gpu(self):
        self._ensure_loaded()
        self.inception.to(self.device)
        self.clip_model.to(self.device)
        self.ae_clip.to(self.device)
        if next(self.ae_clip.parameters()).dtype != torch.float16:
            import clip as openai_clip
            openai_clip.model.convert_weights(self.ae_clip)  # fp16, as clip.load on cuda
        self.ae_mlp.to(self.device)
        self.hps_model.to(self.device)

    def offload(self):
        for m in (self.inception, self.clip_model, self.ae_clip,
                  self.ae_mlp, self.hps_model):
            m.to("cpu")
        torch.cuda.empty_cache()

    # ------------------------------------------------------------ metrics #
    @torch.no_grad()
    def fid_features(self, image_paths: list[str]) -> np.ndarray:
        feats = []
        for i in range(0, len(image_paths), self.batch_size):
            imgs = _load_images(image_paths[i:i + self.batch_size])
            batch = torch.stack([self.fid_tf(im) for im in imgs]).to(self.device)
            feats.append(self.inception(batch).cpu())
        return torch.cat(feats).numpy()

    @torch.no_grad()
    def clip_scores(self, image_paths: list[str], prompts: list[str]) -> np.ndarray:
        scores = []
        for i in range(0, len(image_paths), self.batch_size):
            imgs = _load_images(image_paths[i:i + self.batch_size])
            texts = prompts[i:i + self.batch_size]
            inputs = self.clip_proc(text=texts, images=imgs, return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            img_f = self.clip_model.get_image_features(pixel_values=inputs["pixel_values"])
            txt_f = self.clip_model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            # transformers >=5 returns an output object; the projected features
            # live in .pooler_output (older versions return the tensor directly)
            if not torch.is_tensor(img_f):
                img_f = img_f.pooler_output
            if not torch.is_tensor(txt_f):
                txt_f = txt_f.pooler_output
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            scores.append((img_f * txt_f).sum(-1).cpu())
        return torch.cat(scores).numpy()

    @torch.no_grad()
    def aesthetic_scores(self, image_paths: list[str]) -> np.ndarray:
        scores = []
        for i in range(0, len(image_paths), self.batch_size):
            imgs = _load_images(image_paths[i:i + self.batch_size])
            batch = torch.stack([self.ae_preprocess(im) for im in imgs])
            batch = batch.to(self.device, dtype=next(self.ae_clip.parameters()).dtype)
            emb = self.ae_clip.encode_image(batch)
            emb = _normalized(emb.cpu().float().numpy())
            pred = self.ae_mlp(torch.from_numpy(emb).to(self.device).float())
            scores.append(pred.squeeze(-1).cpu())
        return torch.cat(scores).numpy()

    @torch.no_grad()
    def hps_scores(self, image_paths: list[str], prompts: list[str]) -> np.ndarray:
        scores = []
        for i in range(0, len(image_paths), self.batch_size):
            imgs = _load_images(image_paths[i:i + self.batch_size])
            batch = torch.stack([self.hps_preprocess(im) for im in imgs]).to(self.device)
            text = self.hps_tokenizer(prompts[i:i + self.batch_size]).to(self.device)
            with torch.autocast("cuda"):
                out = self.hps_model(batch, text)
                sim = (out["image_features"] * out["text_features"]).sum(-1)
            scores.append(sim.float().cpu())
        return torch.cat(scores).numpy()

    # ---------------------------------------------------------- one-shot #
    def evaluate(self, image_paths: list[str], prompts: list[str],
                 types: list[str], fid_ref: dict[str, dict]) -> dict:
        """Compute all four metrics. `types[i]` in {'icon','illustration'};
        fid_ref maps type (+ 'all') -> {'mu':..., 'sigma':...}."""
        self.to_gpu()
        try:
            res = {}
            feats = self.fid_features(image_paths)
            for t in ("icon", "illustration"):
                mask = np.array([tt == t for tt in types])
                if mask.any() and t in fid_ref:
                    mu_g, sig_g = activation_statistics(feats[mask])
                    res[f"fid_{t}"] = float(frechet_distance(
                        fid_ref[t]["mu"], fid_ref[t]["sigma"], mu_g, sig_g))
            if "all" in fid_ref:
                mu_g, sig_g = activation_statistics(feats)
                res["fid_all"] = float(frechet_distance(
                    fid_ref["all"]["mu"], fid_ref["all"]["sigma"], mu_g, sig_g))

            for name, arr in (("clip", self.clip_scores(image_paths, prompts)),
                              ("aesthetic", self.aesthetic_scores(image_paths)),
                              ("hps", self.hps_scores(image_paths, prompts))):
                res[name] = float(arr.mean())
                for t in ("icon", "illustration"):
                    mask = np.array([tt == t for tt in types])
                    if mask.any():
                        res[f"{name}_{t}"] = float(arr[mask].mean())

            # reporting headline numbers: mean of the two per-type values
            for m in ("fid", "clip", "aesthetic", "hps"):
                a, b = res.get(f"{m}_icon"), res.get(f"{m}_illustration")
                if a is not None and b is not None:
                    res[f"{m}_mean"] = (a + b) / 2
            return res
        finally:
            self.offload()


def load_fid_reference(ref_dir: str | Path) -> dict[str, dict]:
    """Load precomputed reference stats written by build_fid_ref.py."""
    ref = {}
    for key in ("icon", "illustration", "all"):
        p = Path(ref_dir) / f"{key}.npz"
        if p.exists():
            d = np.load(p)
            ref[key] = {"mu": d["mu"], "sigma": d["sigma"]}
    return ref
