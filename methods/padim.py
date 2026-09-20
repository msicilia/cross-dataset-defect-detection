"""PaDiM: a multivariate Gaussian per patch position, scored by Mahalanobis distance.

    memory-bank detectors              PaDiM
    ---------------------------------  -----------------------------------
    non-parametric, stores vectors     parametric, stores mean + covariance
    score = distance to nearest        score = Mahalanobis distance to the
            stored patch                       fitted distribution

Features as for PatchCore: WideResNet-50 layer2 and layer3 at 28x28 after a
square resize and 224 centre crop, restricted to one random subset of n_dims
channels shared by all positions. Image score = max over positions.

Reference:
    Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly
    Detection and Localization", ICPR 2021.
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


class PaDiM(AnomalyMethod):
    name = "padim"

    def __init__(
        self,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        n_dims: int = 200,
        max_train_images: int = 500,
        batch_size: int = 32,
        grid: int = 28,
        # Covariance regulariser as a fraction of the mean per-channel variance.
        # (Defard et al. use an absolute 0.01.)
        eps: float = 0.01,
        device: str | None = None,
    ):
        self.layers = layers
        self.n_dims = n_dims
        self.max_train_images = max_train_images
        self.batch_size = batch_size
        self.grid = grid
        self.eps = eps
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
        self.mean: torch.Tensor | None = None      # (P, d)
        self.inv_cov: torch.Tensor | None = None   # (P, d, d)
        self.dim_idx: np.ndarray | None = None     # selected channel subset

    def _load_batch(self, paths: list[Path]) -> torch.Tensor:
        return torch.stack(load_all(_load, paths)).to(self.device)

    @torch.no_grad()
    def _patch_features(self, paths: list[Path], desc: str) -> torch.Tensor:
        """(N, P, d) features, P = grid*grid positions, d = selected channels."""
        out = []
        for i in tqdm(range(0, len(paths), self.batch_size), desc=desc, leave=False):
            batch = self._load_batch(paths[i : i + self.batch_size])
            self._features.clear()
            self.backbone(batch)
            feats = [F.adaptive_avg_pool2d(self._features[n], (self.grid, self.grid))
                     for n in self.layers]
            combined = torch.cat(feats, dim=1)                    # (B, C, H, W)
            B, C, H, W = combined.shape
            x = combined.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, P, C)
            out.append(x[:, :, self.dim_idx].cpu())
        return torch.cat(out, dim=0)

    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        paths = image_paths
        if len(paths) > self.max_train_images:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), self.max_train_images, replace=False)
            paths = [paths[i] for i in idx]

        # One channel subset per fit, shared by all positions, drawn from a
        # separate stream from the reference-image draw.
        rng = np.random.default_rng([seed, 1])
        n_channels = sum({"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}[n]
                         for n in self.layers)
        self.dim_idx = np.sort(rng.choice(n_channels, self.n_dims, replace=False))

        print(f"  [padim] extracting features from {len(paths)} images …")
        x = self._patch_features(paths, desc="extracting").double()   # (N, P, d)
        N, P, d = x.shape
        if N <= d:
            raise ValueError(f"PaDiM needs more reference images than channels "
                             f"(got N={N}, n_dims={d}); lower n_dims")

        print(f"  [padim] fitting {P} Gaussians of dimension {d} from {N} images …")
        mean = x.mean(dim=0)                                          # (P, d)
        c = (x - mean).permute(1, 0, 2)                               # (P, N, d)
        cov = torch.einsum("pnd,pne->pde", c, c) / (N - 1)
        scale = torch.diagonal(cov, dim1=1, dim2=2).mean()
        cov += (self.eps * scale) * torch.eye(d, dtype=cov.dtype).unsqueeze(0)
        # Inverted in float64, stored in float32 for the device (MPS has no float64).
        self.inv_cov = torch.linalg.inv(cov).float()
        self.mean = mean.float()

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        assert self.mean is not None, "Call fit() first"
        dev = self.device
        mu, inv = self.mean.to(dev), self.inv_cov.to(dev)
        scores = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"scoring [{self.name}]"):
            x = self._patch_features(image_paths[i : i + self.batch_size], desc="").to(dev)
            delta = x - mu                                            # (B, P, d)
            m = torch.einsum("bpd,pde,bpe->bp", delta, inv, delta)     # (B, P)
            scores.append(m.clamp_min(0).sqrt().max(dim=1).values.cpu().numpy())
        return np.concatenate(scores)
