from __future__ import annotations
"""PatchCore baseline with a WideResNet-50 backbone (supervised ImageNet features).

Identical memory-bank logic to DINOPatchCore; only the feature extractor differs.
This isolates the contribution of the DINOv2 backbone.

Reference:
    Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from .base import AnomalyMethod
from .dino_patchcore import _greedy_coreset


_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PatchCore(AnomalyMethod):
    name = "patchcore"

    def __init__(
        self,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        coreset_ratio: float = 0.01,
        max_train_images: int = 500,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.layers = layers
        self.coreset_ratio = coreset_ratio
        self.max_train_images = max_train_images
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        backbone.eval().to(self.device)
        for p in backbone.parameters():
            p.requires_grad_(False)

        # Register forward hooks to capture intermediate feature maps
        self._features: dict[str, torch.Tensor] = {}
        for layer_name in layers:
            getattr(backbone, layer_name).register_forward_hook(
                lambda m, inp, out, name=layer_name: self._features.__setitem__(name, out)
            )
        self.backbone = backbone
        self.memory_bank: np.ndarray | None = None

    def _load_batch(self, paths: list[Path]) -> torch.Tensor:
        imgs = [_TRANSFORM(Image.open(p).convert("RGB")) for p in paths]
        return torch.stack(imgs).to(self.device)

    @torch.no_grad()
    def _extract(self, paths: list[Path]) -> np.ndarray:
        all_patches = []
        for i in tqdm(range(0, len(paths), self.batch_size), desc="extracting", leave=False):
            batch = self._load_batch(paths[i : i + self.batch_size])
            self._features.clear()
            self.backbone(batch)
            feats = []
            for name in self.layers:
                f = self._features[name]  # (B, C, H, W)
                # adaptive pool to common spatial size (H=28, W=28)
                f = F.adaptive_avg_pool2d(f, output_size=(28, 28))
                feats.append(f)
            combined = torch.cat(feats, dim=1)  # (B, C_total, 28, 28)
            B, C, H, W = combined.shape
            patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C)
            all_patches.append(patches.cpu().numpy())
        return np.concatenate(all_patches, axis=0)

    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        paths = image_paths
        if len(paths) > self.max_train_images:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), self.max_train_images, replace=False)
            paths = [paths[i] for i in idx]

        print(f"  [patchcore] extracting features from {len(paths)} images …")
        features = self._extract(paths)
        target = max(1, int(len(features) * self.coreset_ratio))
        print(f"  [patchcore] building coreset: {len(features)} → {target} …")
        self.memory_bank = _greedy_coreset(features, target, seed=seed, pre_sample=50_000)

    def _nn_distances(self, patches: np.ndarray) -> np.ndarray:
        mb = torch.tensor(self.memory_bank, dtype=torch.float32)
        dists = []
        for i in range(0, len(patches), 4096):
            chunk = torch.tensor(patches[i : i + 4096], dtype=torch.float32)
            dists.append(torch.cdist(chunk, mb).min(dim=1).values.numpy())
        return np.concatenate(dists)

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        assert self.memory_bank is not None
        scores = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="scoring [patchcore]"):
            batch_paths = image_paths[i : i + self.batch_size]
            batch = self._load_batch(batch_paths)
            self._features.clear()
            self.backbone(batch)
            feats = [F.adaptive_avg_pool2d(self._features[n], (28, 28)) for n in self.layers]
            combined = torch.cat(feats, dim=1)  # (B, C, 28, 28)
            B, C, H, W = combined.shape
            patches = combined.permute(0, 2, 3, 1).reshape(B * H * W, C).cpu().numpy()
            dists = self._nn_distances(patches).reshape(B, H * W)
            scores.append(dists.max(axis=1))
        return np.concatenate(scores)
