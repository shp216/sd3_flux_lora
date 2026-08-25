"""FluxLoraDataset: loads (image, caption) pairs. Text is encoded inline at train-time.

Image paths in the manifest may be absolute *or* relative to the manifest's
parent directory -- this makes the manifest portable across hosts.
"""
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FluxLoraDataset(Dataset):
    def __init__(self, manifest_path: str, resolution: int = 1024):
        self.manifest_dir = Path(manifest_path).resolve().parent
        with open(manifest_path) as f:
            self.items = [json.loads(line) for line in f if line.strip()]
        self.image_tf = transforms.Compose([
            transforms.Resize(resolution, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # -> [-1, 1]
        ])

    def _resolve(self, img_path: str) -> Path:
        p = Path(img_path)
        return p if p.is_absolute() else self.manifest_dir / p

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        img = Image.open(self._resolve(it["image"])).convert("RGB")
        return {
            "pixel_values": self.image_tf(img),
            "caption":      it["caption"],
        }


def collate(batch):
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "captions":     [b["caption"] for b in batch],
    }
