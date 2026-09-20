"""SPADE: k-nearest-neighbour retrieval of whole reference images.

    PatchCore                          SPADE
    ---------------------------------  -----------------------------------
    stores a coreset of patch vectors  stores one descriptor per image
    greedy farthest-point subset: a    keeps every drawn reference image
    coreset_ratio fraction of all
    reference patches
    nearest neighbour over patches     k nearest neighbours over images
    image score = max over patches     image score = mean over k neighbours

Backbone: WideResNet-50 with ImageNet weights, as for PatchCore. The image
descriptor concatenates global average pools of layer2, layer3 and layer4.

Reference:
    Cohen and Hoshen, "Sub-Image Anomaly Detection with Deep Pyramid
    Correspondences", arXiv:2005.02357, 2020.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from .base import AnomalyMethod
from .imageio import load_all

# Square resize before the centre crop, as for PatchCore and DINO-PatchCore.
_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load(path: Path) -> torch.Tensor:
    return _TRANSFORM(Image.open(path).convert("RGB"))


class SPADE(AnomalyMethod):
    name = "spade"

    def __init__(
        self,
        # layer4 is included because SPADE compares whole images, for which
        # Cohen and Hoshen use deep features.
        layers: tuple[str, ...] = ("layer2", "layer3", "layer4"),
        k: int = 50,
        max_train_images: int = 500,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.layers = layers
        self.k = k
        self.max_train_images = max_train_images
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")

        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval().to(self.device)
        for p in backbone.parameters():
            p.requires_grad_(False)

        self._features: dict[str, torch.Tensor] = {}
        for layer_name in layers:
            getattr(backbone, layer_name).register_forward_hook(
                lambda m, inp, out, name=layer_name: self._features.__setitem__(name, out)
            )
        self.backbone = backbone
        self.gallery: np.ndarray | None = None   # (N_ref, C) image descriptors

    def _load_batch(self, paths: list[Path]) -> torch.Tensor:
        return torch.stack(load_all(_load, paths)).to(self.device)

    @torch.no_grad()
    def _descriptors(self, paths: list[Path], desc: str) -> np.ndarray:
        """One globally average-pooled descriptor per image."""
        out = []
        for i in tqdm(range(0, len(paths), self.batch_size), desc=desc, leave=False):
            batch = self._load_batch(paths[i : i + self.batch_size])
            self._features.clear()
            self.backbone(batch)
            pooled = [F.adaptive_avg_pool2d(self._features[n], (1, 1)).flatten(1)
                      for n in self.layers]
            out.append(torch.cat(pooled, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        paths = image_paths
        if len(paths) > self.max_train_images:
            # Same draw as the other detectors, so all see the same reference
            # images for a given seed.
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), self.max_train_images, replace=False)
            paths = [paths[i] for i in idx]
        print(f"  [spade] building gallery from {len(paths)} images …")
        self.gallery = self._descriptors(paths, desc="gallery [spade]")

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        assert self.gallery is not None, "Call fit() first"
        gal = torch.tensor(self.gallery, dtype=torch.float32)
        k = min(self.k, gal.shape[0])
        scores = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"scoring [{self.name}]"):
            q = torch.tensor(self._descriptors(image_paths[i : i + self.batch_size], desc=""),
                             dtype=torch.float32)
            d = torch.cdist(q, gal)                     # (B, N_ref)
            scores.append(d.topk(k, largest=False).values.mean(dim=1).numpy())
        return np.concatenate(scores)
