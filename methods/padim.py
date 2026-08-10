from __future__ import annotations
"""PaDiM: per-position Gaussian modelling of normal patch features.

This is the control that distinguishes two very different readings of our
central result. PatchCore and DINO-PatchCore share a scoring rule -- nearest
neighbour against stored exemplars -- so their near-identical generalisation gap
can only speak to backbone choice. PaDiM stores *no exemplars at all*: it fits a
multivariate Gaussian to each patch position and scores by Mahalanobis distance.

    memory-bank family                 PaDiM
    ---------------------------------  -----------------------------------
    non-parametric, stores vectors     parametric, stores mean + covariance
    score = distance to nearest        score = Mahalanobis distance to the
            stored patch                       fitted distribution
    no notion of feature covariance    covariance is the whole model

If PaDiM's gap resembles the memory-bank methods', the bottleneck is not the
memory-bank paradigm but cross-domain transfer of unsupervised anomaly detection
generally. If its gap is smaller, the memory-bank reading is supported. Either
outcome is informative, which is why it is worth running.

The backbone (WideResNet-50) and the layers, pooling resolution and reference
subsampling are identical to PatchCore's, so the representation is held fixed.

Reference:
    Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly
    Detection and Localization", ICPR 2021.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from .base import AnomalyMethod

_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class PaDiM(AnomalyMethod):
    name = "padim"

    def __init__(
        self,
        layers: tuple[str, ...] = ("layer2", "layer3"),
        n_dims: int = 200,
        max_train_images: int = 500,
        batch_size: int = 32,
        grid: int = 28,
        # Regularisation is RELATIVE to the feature scale, unlike the paper's
        # absolute 0.01. Our pooled WideResNet features have a mean per-dimension
        # variance of ~0.01, so an absolute 0.01 equals the entire signal and
        # 68% of dimensions would have variance below it -- the covariance would
        # be swamped and PaDiM would degenerate towards Euclidean distance,
        # which is precisely the model it is meant to improve on. 1% of the mean
        # variance keeps the covariance meaningful; the resulting condition
        # number (~5e3) was verified to be harmless in float32 (max relative
        # error 5e-7, rank correlation 1.0 against a float64 reference), which
        # matters because MPS does not support float64.
        eps: float = 0.01,
        device: str | None = None,
    ):
        # n_dims must stay well below the number of reference images, otherwise
        # the per-position sample covariance is rank-deficient. Our protocol caps
        # references at 500, so the paper's d=550 is not estimable here; d=200
        # keeps N/d = 2.5 while retaining most of the original's capacity.
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
        imgs = [_TRANSFORM(Image.open(p).convert("RGB")) for p in paths]
        return torch.stack(imgs).to(self.device)

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
            combined = torch.cat(feats, dim=1)                    # (B, C, G, G)
            B, C, G, W = combined.shape
            x = combined.permute(0, 2, 3, 1).reshape(B, G * W, C)  # (B, P, C)
            out.append(x[:, :, self.dim_idx].cpu())
        return torch.cat(out, dim=0)

    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        paths = image_paths
        if len(paths) > self.max_train_images:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), self.max_train_images, replace=False)
            paths = [paths[i] for i in idx]

        # Channel subset is drawn once per fit, seeded, as in the original.
        rng = np.random.default_rng(seed)
        n_channels = sum({"layer1": 256, "layer2": 512, "layer3": 1024, "layer4": 2048}[n]
                         for n in self.layers)
        self.dim_idx = np.sort(rng.choice(n_channels, self.n_dims, replace=False))

        print(f"  [padim] extracting features from {len(paths)} images …")
        x = self._patch_features(paths, desc="extracting").double()   # (N, P, d)
        N, P, d = x.shape
        if N <= d:
            raise ValueError(f"PaDiM needs more reference images than dimensions "
                             f"(got N={N}, d={d}); lower n_dims")

        print(f"  [padim] fitting {P} Gaussians of dimension {d} from {N} images …")
        mean = x.mean(dim=0)                                          # (P, d)
        c = (x - mean).permute(1, 0, 2)                               # (P, N, d)
        cov = torch.einsum("pnd,pne->pde", c, c) / (N - 1)
        # Shrink towards the identity by a fraction of the mean feature variance
        # (see the note on eps above): scale-free, so it behaves the same
        # whatever the backbone's activation magnitude happens to be.
        scale = torch.diagonal(cov, dim1=1, dim2=2).mean()
        cov += (self.eps * scale) * torch.eye(d, dtype=cov.dtype).unsqueeze(0)
        self.inv_cov = torch.linalg.inv(cov).float()
        self.mean = mean.float()

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        assert self.mean is not None, "Call fit() first"
        # Mahalanobis is a batched quadratic form; run it on the accelerator.
        dev = self.device
        mu, inv = self.mean.to(dev), self.inv_cov.to(dev)
        scores = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"scoring [{self.name}]"):
            x = self._patch_features(image_paths[i : i + self.batch_size], desc="").to(dev)
            delta = x - mu                                            # (B, P, d)
            m = torch.einsum("bpd,pde,bpe->bp", delta, inv, delta)     # (B, P)
            # Image score is the worst position, matching PatchCore's max rule.
            scores.append(m.clamp_min(0).sqrt().max(dim=1).values.cpu().numpy())
        return np.concatenate(scores)
