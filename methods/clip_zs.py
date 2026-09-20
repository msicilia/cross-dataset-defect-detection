"""CLIP zero-shot defect scoring (CLIP-ZS).

Each image is resized to image_size x image_size and encoded once by CLIP.
For every (positive, negative) prompt pair, the positive-class probability is
the softmax over the two scaled cosine similarities, using CLIP's learned
logit scale. The anomaly score is the maximum of these probabilities over all
pairs. No reference images are used.

Reference:
    Radford et al., "Learning Transferable Visual Models From Natural
    Language Supervision", ICML 2021.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .base import AnomalyMethod
from .imageio import load_all


class CLIPZS(AnomalyMethod):
    name = "clip_zs"

    def __init__(
        self,
        prompts: dict[str, list[tuple[str, str]]],
        model_id: str = "openai/clip-vit-large-patch14",
        image_size: int = 224,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.prompts = prompts
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._encode_prompts()

    @torch.no_grad()
    def _encode_prompts(self) -> None:
        pairs = [pair for group in self.prompts.values() for pair in group]

        def encode(texts: list[str]) -> torch.Tensor:
            inputs = self.processor(text=texts, return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            return feats / feats.norm(dim=-1, keepdim=True)

        self._pos_feats = encode([pos for pos, _ in pairs])   # (K, D)
        self._neg_feats = encode([neg for _, neg in pairs])   # (K, D)

    def _load_image(self, path: Path) -> Image.Image:
        return Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)

    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        pass

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        all_scores = []
        logit_scale = self.model.logit_scale.exp()
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="scoring [clip_zs]"):
            images = load_all(self._load_image, image_paths[i : i + self.batch_size])
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            img_feats = self.model.get_image_features(**inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)   # (B, D)

            sim_pos = logit_scale * (img_feats @ self._pos_feats.T)        # (B, K)
            sim_neg = logit_scale * (img_feats @ self._neg_feats.T)
            # Two-way softmax over a pair, written as a sigmoid of the difference.
            pair_scores = torch.sigmoid(sim_pos - sim_neg)
            all_scores.append(pair_scores.max(dim=1).values.cpu().numpy())

        return np.concatenate(all_scores)
