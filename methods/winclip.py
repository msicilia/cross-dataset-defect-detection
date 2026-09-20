"""WinCLIP (zero-shot) and WinCLIP+ (with reference images).

Windows. Each image is resized to image_size x image_size. Its windows are the
full image plus, for each k in `scales` (2 and 3), the k x k non-overlapping
tiles of side image_size // k (when image_size is not divisible by k, the last
image_size mod k pixel rows and columns are not covered at that scale). With
the default scales an image yields 1 + 4 + 9 = 14 windows. Every window is
passed separately through the CLIP processor, so tiles are upsampled to the
processor resolution and re-encoded by the CLIP image encoder; embeddings are
L2-normalised.

Language score. For each window and each (positive, negative) prompt pair, the
positive-class probability is the softmax over the two cosine similarities
scaled by CLIP's learned logit scale; the window's language score is the
maximum over pairs. The prompt pairs are those of CLIP-ZS.

WinCLIP+ adds a reference term. fit() encodes the windows of up to
max_train_images reference images (drawn with the run seed) at all scales into
one set. A test window's reference term is its Euclidean distance to the
nearest reference window of any scale, divided by 2 (the maximum distance
between unit vectors) and clipped to [0, 1]; its score is
0.5 * language score + 0.5 * reference term. WinCLIP uses the language score
alone.

The image score is the maximum window score.

Differences from Jeong et al.: the prompt pairs replace the compositional
prompt ensemble, and windows are whole-image crops re-encoded by CLIP rather
than masked token subsets of one forward pass.

Reference:
    Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and
    Segmentation", CVPR 2023.
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


class WinCLIP(AnomalyMethod):
    def __init__(
        self,
        prompts: dict[str, list[tuple[str, str]]],
        model_id: str = "openai/clip-vit-large-patch14",
        image_size: int = 224,
        scales: tuple[int, ...] = (2, 3),
        few_shot: bool = False,
        max_train_images: int = 500,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.name = "winclip_plus" if few_shot else "winclip"
        self.model_id = model_id
        self.prompts = prompts
        self.image_size = image_size
        self.scales = scales
        self.few_shot = few_shot
        self.max_train_images = max_train_images
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.temperature = float(self.model.logit_scale.exp().item())
        self.reference: np.ndarray | None = None   # (N * windows, D)
        self._pos = self._neg = None

    @torch.no_grad()
    def _encode_prompts(self) -> None:
        pairs = [pair for group in self.prompts.values() for pair in group]

        def enc(texts):
            inp = self.processor(text=texts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
            f = self.model.get_text_features(**inp)
            return f / f.norm(dim=-1, keepdim=True)

        self._pos = enc([pos for pos, _ in pairs])
        self._neg = enc([neg for _, neg in pairs])

    def _windows(self, img: Image.Image) -> list[Image.Image]:
        out = [img]
        w, h = img.size
        for k in self.scales:
            sw, sh = w // k, h // k
            for i in range(k):
                for j in range(k):
                    out.append(img.crop((j * sw, i * sh, (j + 1) * sw, (i + 1) * sh)))
        return out

    def _load_windows(self, path: Path) -> list[Image.Image]:
        img = Image.open(path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR)
        return self._windows(img)

    @torch.no_grad()
    def _window_features(self, paths: list[Path], desc: str) -> tuple[np.ndarray, int]:
        """(len(paths) * W, D) window embeddings, grouped by image, and W."""
        feats, n_win = [], None
        for i in tqdm(range(0, len(paths), self.batch_size), desc=desc, leave=False):
            per_image = load_all(self._load_windows, paths[i : i + self.batch_size])
            n_win = len(per_image[-1])
            batch_windows = [w for windows in per_image for w in windows]
            inp = self.processor(images=batch_windows, return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0), n_win

    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        if self._pos is None:
            self._encode_prompts()
        if not self.few_shot:
            return
        paths = list(image_paths)
        if len(paths) > self.max_train_images:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), self.max_train_images, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        print(f"  [{self.name}] encoding {len(paths)} reference images")
        self.reference, _ = self._window_features(paths, desc=f"reference [{self.name}]")

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        if self.few_shot and self.reference is None:
            raise RuntimeError(f"{self.name}: fit() must be called before score()")
        if self._pos is None:
            self._encode_prompts()
        ref = torch.tensor(self.reference) if self.few_shot else None
        scores = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"scoring [{self.name}]"):
            chunk = image_paths[i : i + self.batch_size]
            f, n_win = self._window_features(chunk, desc="")
            t = torch.tensor(f).to(self.device)
            sp = t @ self._pos.T                       # (N * W, K)
            sn = t @ self._neg.T
            lang = torch.softmax(torch.stack([sp, sn], -1) * self.temperature,
                                 -1)[..., 0].max(-1).values
            s = lang.reshape(len(chunk), n_win).cpu()
            if ref is not None:
                d = torch.cdist(t.cpu(), ref).min(dim=1).values.reshape(len(chunk), n_win)
                s = 0.5 * s + 0.5 * (d / 2.0).clamp(0, 1)
            scores.append(s.max(dim=1).values.numpy())
        return np.concatenate(scores)
