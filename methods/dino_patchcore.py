from __future__ import annotations
"""DINO-PatchCore: PatchCore anomaly detection with a frozen DINOv2 backbone.

Memory bank is built from patch tokens of defect-free training images.
Image-level score: max nearest-neighbour distance across all image patches.
Pixel-level map: per-patch NN distance, upsampled to input resolution.

Reference:
    Roth et al., "Towards Total Recall in Industrial Anomaly Detection",
    CVPR 2022 (PatchCore).
    Oquab et al., "DINOv2", TMLR 2023.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from .base import AnomalyMethod


def _greedy_coreset(
    features: np.ndarray,
    target_size: int,
    seed: int = 0,
    pre_sample: int | None = None,
) -> np.ndarray:
    """Greedy farthest-point sampling with optional random pre-sampling.

    Pass pre_sample to bound O(n * target_size) complexity when n is large
    (e.g., PatchCore with 400k ResNet patches). Leave None to use all features.
    """
    rng = np.random.default_rng(seed)
    n = len(features)
    if pre_sample is not None and n > pre_sample:
        idx = rng.choice(n, pre_sample, replace=False)
        features = features[idx]
        n = pre_sample
    target_size = min(target_size, n)
    selected = [int(rng.integers(n))]
    min_dists = np.full(n, np.inf, dtype=np.float32)
    for _ in range(target_size - 1):
        d = np.linalg.norm(features - features[selected[-1]], axis=1).astype(np.float32)
        np.minimum(min_dists, d, out=min_dists)
        selected.append(int(np.argmax(min_dists)))
    return features[selected]


class DINOPatchCore(AnomalyMethod):

    def __init__(
        self,
        backbone: str = "facebook/dinov2-base",
        coreset_ratio: float = 0.01,
        max_train_images: int = 500,
        image_size: int = 256,
        batch_size: int = 16,
        device: str | None = None,
        # Optional image-space normalisation applied identically to reference
        # and test images, used by the confounder ablations (run_confounders.py)
        # to strip one candidate explanation at a time -- colour, illumination,
        # or resolution -- and see whether the transfer gap survives without it.
        # None reproduces the standard pipeline exactly.
        preprocess=None,
        variant: str = "",
    ):
        self.preprocess = preprocess
        self.name = f"dino_patchcore_{backbone.split('/')[-1]}" + (f"_{variant}" if variant else "")
        self.backbone_id = backbone
        self.coreset_ratio = coreset_ratio
        self.max_train_images = max_train_images
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.processor = AutoImageProcessor.from_pretrained(backbone)
        self.model = AutoModel.from_pretrained(backbone).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.memory_bank: np.ndarray | None = None
        self._patch_hw: tuple[int, int] | None = None  # (H_patches, W_patches)

    # ── feature extraction ────────────────────────────────────────────────────

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        # Applied after resizing so every variant sees the same geometry, and to
        # both reference and test images so the two are never mismatched.
        return self.preprocess(img) if self.preprocess is not None else img

    @torch.no_grad()
    def _extract_patches(self, paths: list[Path]) -> np.ndarray:
        """Returns array of shape (N_patches_total, D)."""
        all_patches = []
        for i in tqdm(range(0, len(paths), self.batch_size), desc="extracting", leave=False):
            batch_paths = paths[i : i + self.batch_size]
            images = [self._load_image(p) for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            # patch tokens: last_hidden_state[:, 1:, :] (skip CLS token)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, P, D)
            B, P, D = patch_tokens.shape
            # remember spatial layout from first batch
            if self._patch_hw is None:
                side = int(P ** 0.5)
                self._patch_hw = (side, side)
            all_patches.append(patch_tokens.reshape(B * P, D).cpu().numpy())
        return np.concatenate(all_patches, axis=0)

    @torch.no_grad()
    def _extract_patch_maps(self, paths: list[Path]) -> list[np.ndarray]:
        """Returns list of (H_p, W_p, D) feature maps, one per image."""
        maps = []
        for i in tqdm(range(0, len(paths), self.batch_size),
                      desc=f"score_maps [{self.name}]", leave=False):
            batch_paths = paths[i : i + self.batch_size]
            images = [self._load_image(p) for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, P, D)
            B, P, D = patch_tokens.shape
            side = int(P ** 0.5)
            for b in range(B):
                maps.append(patch_tokens[b].reshape(side, side, D).cpu().numpy())
        return maps

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        paths = image_paths
        if len(paths) > self.max_train_images:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), self.max_train_images, replace=False)
            paths = [paths[i] for i in idx]

        print(f"  [{self.name}] extracting features from {len(paths)} training images …")
        features = self._extract_patches(paths)

        target = max(1, int(len(features) * self.coreset_ratio))
        print(f"  [{self.name}] building coreset: {len(features)} → {target} vectors …")
        self.memory_bank = _greedy_coreset(features, target, seed=seed)
        print(f"  [{self.name}] memory bank ready: {self.memory_bank.shape}")

    # ── score ─────────────────────────────────────────────────────────────────

    def _nn_distances(self, patch_features: np.ndarray) -> np.ndarray:
        """Compute nearest-neighbour distance of each patch to the memory bank."""
        mb = torch.tensor(self.memory_bank, dtype=torch.float32)  # (M, D)
        chunk_size = 2048
        dists = []
        for i in range(0, len(patch_features), chunk_size):
            chunk = torch.tensor(patch_features[i : i + chunk_size], dtype=torch.float32)
            d = torch.cdist(chunk, mb).min(dim=1).values
            dists.append(d.numpy())
        return np.concatenate(dists)

    def score(self, image_paths: list[Path]) -> np.ndarray:
        assert self.memory_bank is not None, "Call fit() first"
        scores = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"scoring [{self.name}]"):
            batch = image_paths[i : i + self.batch_size]
            images = [self._load_image(p) for p in batch]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            patches = outputs.last_hidden_state[:, 1:, :]  # (B, P, D)
            B, P, D = patches.shape
            flat = patches.reshape(B * P, D).cpu().numpy()
            dists = self._nn_distances(flat).reshape(B, P)
            scores.append(dists.max(axis=1))  # image score = max patch distance
        return np.concatenate(scores)

    def score_maps(self, image_paths: list[Path]) -> list[np.ndarray]:
        assert self.memory_bank is not None, "Call fit() first"
        # _extract_patch_maps slices its argument and calls len() on it, so it
        # needs a real sequence rather than an iterator. Progress is reported
        # inside the extractor.
        patch_maps = self._extract_patch_maps(list(image_paths))
        result = []
        for fmap in patch_maps:
            H, W, D = fmap.shape
            flat = fmap.reshape(H * W, D)
            dists = self._nn_distances(flat).reshape(H, W)
            # bilinear upsample to image_size
            t = torch.tensor(dists[None, None]).float()
            t = F.interpolate(t, size=(self.image_size, self.image_size), mode="bilinear",
                              align_corners=False)
            result.append(t[0, 0].numpy())
        return result
