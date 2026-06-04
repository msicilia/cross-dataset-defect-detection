from __future__ import annotations
"""CLIP zero-shot defect classification (CLIP-ZS).

For each image, computes softmax probability of belonging to each positive
prompt versus its paired negative prompt.  The final anomaly score is the
maximum positive probability across all defect categories.

No training data is required; the method is dataset-agnostic.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .base import AnomalyMethod


class CLIPZS(AnomalyMethod):
    name = "clip_zs"

    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        prompts: dict[str, list[tuple[str, str]]] | None = None,
        image_size: int = 224,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Default prompt pairs (positive, negative) per defect category
        self.prompts: dict[str, list[tuple[str, str]]] = prompts or {
            "crack": [
                ("a photo of a concrete wall with a crack",
                 "a photo of a smooth intact concrete wall"),
                ("structural damage: crack on a building surface",
                 "undamaged building surface with no cracks"),
                ("a hairline crack on concrete",
                 "pristine concrete surface"),
            ],
            "spalling": [
                ("concrete surface with spalling and exposed aggregate",
                 "smooth intact concrete surface"),
                ("deteriorated concrete with chunks falling off",
                 "well-maintained concrete wall"),
            ],
            "corrosion": [
                ("a corroded metal surface with rust",
                 "a clean intact metal surface without rust"),
                ("rusty metal component with orange corrosion stains",
                 "unpainted metal surface in good condition"),
            ],
            "staining": [
                ("concrete wall with humidity stains and white efflorescence",
                 "dry clean concrete wall with no staining"),
            ],
        }

        # Pre-encode all text prompts
        self._pos_feats: list[torch.Tensor] = []  # shape (num_pairs, D)
        self._neg_feats: list[torch.Tensor] = []
        self._encode_prompts()

    @torch.no_grad()
    def _encode_prompts(self) -> None:
        pos_texts, neg_texts = [], []
        for pairs in self.prompts.values():
            for pos, neg in pairs:
                pos_texts.append(pos)
                neg_texts.append(neg)

        def encode(texts: list[str]) -> torch.Tensor:
            inputs = self.processor(text=texts, return_tensors="pt",
                                    padding=True, truncation=True).to(self.device)
            feats = self.model.get_text_features(**inputs)
            return feats / feats.norm(dim=-1, keepdim=True)

        self._pos_feats = encode(pos_texts)   # (K, D)
        self._neg_feats = encode(neg_texts)   # (K, D)

    # CLIP-ZS requires no fitting; this is a no-op
    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        pass

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        all_scores = []
        logit_scale = self.model.logit_scale.exp()
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc="scoring [clip_zs]"):
            batch_paths = image_paths[i : i + self.batch_size]
            images = [
                Image.open(p).convert("RGB").resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
                for p in batch_paths
            ]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            img_feats = self.model.get_image_features(**inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)  # (B, D)

            # Similarity to each positive/negative prompt pair
            sim_pos = logit_scale * (img_feats @ self._pos_feats.T)  # (B, K)
            sim_neg = logit_scale * (img_feats @ self._neg_feats.T)  # (B, K)

            # Per-pair probability that the image belongs to the positive class
            pair_scores = torch.sigmoid(sim_pos - sim_neg)  # (B, K)

            # Image score: max across all prompt pairs
            image_score = pair_scores.max(dim=1).values.cpu().numpy()
            all_scores.append(image_score)

        return np.concatenate(all_scores)

    def score_per_category(self, image_paths: list[Path]) -> dict[str, np.ndarray]:
        """Return per-defect-category scores (for ablation analysis)."""
        category_scores: dict[str, list] = {cat: [] for cat in self.prompts}
        pair_idx = 0
        cat_ranges: dict[str, range] = {}
        for cat, pairs in self.prompts.items():
            cat_ranges[cat] = range(pair_idx, pair_idx + len(pairs))
            pair_idx += len(pairs)

        logit_scale = self.model.logit_scale.exp()
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            images = [Image.open(p).convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR) for p in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                img_feats = self.model.get_image_features(**inputs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                sim_pos = logit_scale * (img_feats @ self._pos_feats.T)
                sim_neg = logit_scale * (img_feats @ self._neg_feats.T)
                pair_scores = torch.sigmoid(sim_pos - sim_neg).cpu().numpy()
            for cat, rng in cat_ranges.items():
                cat_scores = pair_scores[:, list(rng)].max(axis=1)
                category_scores[cat].append(cat_scores)

        return {cat: np.concatenate(v) for cat, v in category_scores.items()}
