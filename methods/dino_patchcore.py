"""DINO-PatchCore: PatchCore anomaly detection with a frozen DINOv2 backbone.

Memory bank: greedy farthest-point coreset of the patch tokens of defect-free
reference images. Image score: max over patches of the distance to the nearest
coreset vector. Pixel map: per-patch distance, upsampled over the centre crop
the backbone sees (map_box).

References:
    Roth et al., "Towards Total Recall in Industrial Anomaly Detection",
    CVPR 2022 (PatchCore).
    Oquab et al., "DINOv2", TMLR 2023.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from .base import AnomalyMethod
from .imageio import load_all

_CORESET_CHUNK = 16384


def _greedy_coreset(features: np.ndarray, target_size: int, seed: int = 0) -> np.ndarray:
    """Greedy farthest-point (k-center) selection over all features.

    Distances to each new centre are updated in row chunks on a thread pool.
    Every row's norm is computed exactly as over the whole array, so the
    selection does not depend on the chunking or the number of threads.
    """
    n = len(features)
    if n == 0 or target_size < 1:
        raise ValueError(f"coreset needs features and target_size >= 1 "
                         f"(got {n} features, target_size={target_size})")
    rng = np.random.default_rng(seed)
    target_size = min(target_size, n)
    selected = [int(rng.integers(n))]
    min_dists = np.full(n, np.inf, dtype=np.float32)
    bounds = [(s, min(s + _CORESET_CHUNK, n)) for s in range(0, n, _CORESET_CHUNK)]

    def update(bound, centre):
        s, e = bound
        d = np.linalg.norm(features[s:e] - centre, axis=1).astype(np.float32)
        np.minimum(min_dists[s:e], d, out=min_dists[s:e])

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        for _ in range(target_size - 1):
            centre = features[selected[-1]]
            list(pool.map(lambda b: update(b, centre), bounds))
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
        # Optional transform applied after resizing, identically to reference
        # and test images (run_confounders.py).
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

        # The slow processor is the one these checkpoints ship with; pinning it
        # keeps preprocessing fixed if the transformers default changes.
        self.processor = AutoImageProcessor.from_pretrained(backbone, use_fast=False)
        self.model = AutoModel.from_pretrained(backbone).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.memory_bank: np.ndarray | None = None

    # ── feature extraction ────────────────────────────────────────────────────

    def _load_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        return self.preprocess(img) if self.preprocess is not None else img

    @torch.no_grad()
    def _extract_patches(self, paths: list[Path]) -> np.ndarray:
        """Returns array of shape (N_patches_total, D)."""
        all_patches = []
        for i in tqdm(range(0, len(paths), self.batch_size), desc="extracting", leave=False):
            images = load_all(self._load_image, paths[i : i + self.batch_size])
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            patch_tokens = outputs.last_hidden_state[:, 1:, :]  # (B, P, D), CLS dropped
            B, P, D = patch_tokens.shape
            all_patches.append(patch_tokens.reshape(B * P, D).cpu().numpy())
        return np.concatenate(all_patches, axis=0)

    @torch.no_grad()
    def _extract_patch_maps(self, paths: list[Path]) -> list[np.ndarray]:
        """Returns list of (H_p, W_p, D) feature maps, one per image."""
        maps = []
        for i in tqdm(range(0, len(paths), self.batch_size),
                      desc=f"score_maps [{self.name}]", leave=False):
            images = load_all(self._load_image, paths[i : i + self.batch_size])
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
        """Distance of each patch to its nearest memory-bank vector."""
        mb = torch.tensor(self.memory_bank, dtype=torch.float32)  # (M, D)
        chunk_size = 2048
        dists = []
        for i in range(0, len(patch_features), chunk_size):
            chunk = torch.tensor(patch_features[i : i + chunk_size], dtype=torch.float32)
            d = torch.cdist(chunk, mb).min(dim=1).values
            dists.append(d.numpy())
        return np.concatenate(dists)

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        assert self.memory_bank is not None, "Call fit() first"
        scores = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"scoring [{self.name}]"):
            images = load_all(self._load_image, image_paths[i : i + self.batch_size])
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            patches = outputs.last_hidden_state[:, 1:, :]  # (B, P, D)
            B, P, D = patches.shape
            flat = patches.reshape(B * P, D).cpu().numpy()
            dists = self._nn_distances(flat).reshape(B, P)
            scores.append(dists.max(axis=1))
        return np.concatenate(scores)

    def map_box(self) -> tuple[int, int, int]:
        """(top, left, size) of the region the patch grid covers, in the
        image_size x image_size frame produced by _load_image.

        The DINOv2 processor resizes the shortest edge and centre-crops (256 to
        224 for the public checkpoints), so masks must be cropped to this box.
        """
        p = self.processor
        side = (p.size.get("shortest_edge") or p.size.get("height")) if p.do_resize \
            else self.image_size
        crop = p.crop_size["height"] if p.do_center_crop else side
        scale = self.image_size / side
        offset = int(round(((side - crop) // 2) * scale))
        return offset, offset, int(round(crop * scale))

    def score_maps(self, image_paths: list[Path]) -> list[np.ndarray]:
        """Per-pixel maps covering map_box(), not the full loaded image."""
        assert self.memory_bank is not None, "Call fit() first"
        patch_maps = self._extract_patch_maps(list(image_paths))
        _, _, size = self.map_box()
        result = []
        for fmap in patch_maps:
            H, W, D = fmap.shape
            flat = fmap.reshape(H * W, D)
            dists = self._nn_distances(flat).reshape(H, W)
            t = torch.tensor(dists[None, None]).float()
            t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
            result.append(t[0, 0].numpy())
        return result
