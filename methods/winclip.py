from __future__ import annotations
"""WinCLIP and WinCLIP+ : window-based CLIP anomaly detection.

Two purposes.

(1) A recent transformer-based foundation-model detector: open, reproducible
    from public checkpoints, and producing a scalar score compatible with the
    rest of the benchmark.

(2) More importantly, the cleanest available test of this paper's central claim.
    The generalisation gap tracks *dependence on source-domain reference images*
    rather than the memory-bank paradigm, which calls for a contrast in which
    reference-dependence is the only thing that varies. WinCLIP provides exactly
    that:

        WinCLIP   zero-shot   language only, no reference images  -> expect no gap
        WinCLIP+  few-shot    adds normal reference window features -> expect a gap

    Same backbone, same windows, same prompts; the sole difference is whether
    normality is modelled from the source domain. Any gap difference between the
    two cannot be attributed to architecture or representation.

Implementation notes (stated plainly, since they are simplifications):
  - The original aggregates a compositional prompt ensemble over state words and
    templates. We reuse the paper's existing positive/negative prompt pairs so
    that WinCLIP and CLIP-ZS remain directly comparable; the window mechanism,
    which is what distinguishes WinCLIP, is implemented faithfully.
  - Windows are square, multi-scale, and scored by the same softmax over
    positive/negative prompt similarity used by CLIP-ZS. The image score is the
    maximum over windows, mirroring the max-over-patches rule used elsewhere in
    this benchmark.
  - WinCLIP+ combines the language score with the distance to the nearest normal
    reference window feature, as in the original's few-shot variant.

Reference:
    Jeong et al., "WinCLIP: Zero-/Few-Shot Anomaly Classification and
    Segmentation", CVPR 2023.
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from .base import AnomalyMethod


class WinCLIP(AnomalyMethod):
    def __init__(
        self,
        model_id: str = "openai/clip-vit-large-patch14",
        prompts: dict[str, list[tuple[str, str]]] | None = None,
        image_size: int = 224,
        scales: tuple[int, ...] = (2, 3),   # k x k window grids, plus the full image
        few_shot: bool = False,
        max_train_images: int = 500,
        batch_size: int = 32,
        device: str | None = None,
    ):
        self.name = "winclip_plus" if few_shot else "winclip"
        self.model_id = model_id
        self.image_size = image_size
        self.scales = scales
        self.few_shot = few_shot
        self.max_train_images = max_train_images
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        if not prompts:
            raise ValueError(
                "WinCLIP needs the same prompt pairs as CLIP-ZS so the two are "
                "comparable; pass cfg.CLIP_ZS['prompts']")
        self.prompts = prompts
        # CLIP's learned temperature, matching CLIP-ZS rather than hard-coding 100.
        self.temperature = float(self.model.logit_scale.exp().item())
        self.reference: np.ndarray | None = None   # (N, D) normal window features
        self._pos = self._neg = None

    # ── prompts ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_prompts(self) -> None:
        pos, neg = [], []
        for pairs in self.prompts.values():
            for p, n in pairs:
                pos.append(p); neg.append(n)

        def enc(texts):
            inp = self.processor(text=texts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
            f = self.model.get_text_features(**inp)
            return f / f.norm(dim=-1, keepdim=True)

        self._pos, self._neg = enc(pos), enc(neg)

    # ── windows ───────────────────────────────────────────────────────────────

    def _windows(self, img: Image.Image) -> list[Image.Image]:
        """Full image plus every k x k tile, for each scale."""
        out = [img]
        w, h = img.size
        for k in self.scales:
            sw, sh = w // k, h // k
            for i in range(k):
                for j in range(k):
                    out.append(img.crop((j * sw, i * sh, (j + 1) * sw, (i + 1) * sh)))
        return out

    @torch.no_grad()
    def _window_features(self, paths: list[Path], desc: str) -> tuple[np.ndarray, int]:
        """(N*W, D) L2-normalised window embeddings and the window count W."""
        feats, n_win = [], None
        for i in tqdm(range(0, len(paths), self.batch_size), desc=desc, leave=False):
            batch_windows, counts = [], []
            for p in paths[i : i + self.batch_size]:
                img = Image.open(p).convert("RGB").resize(
                    (self.image_size, self.image_size), Image.BILINEAR)
                ws = self._windows(img)
                batch_windows.extend(ws); counts.append(len(ws))
            n_win = counts[0]
            inp = self.processor(images=batch_windows, return_tensors="pt").to(self.device)
            f = self.model.get_image_features(**inp)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0), n_win

    # ── API ───────────────────────────────────────────────────────────────────

    def fit(self, image_paths: list[Path], seed: int = 0) -> None:
        if self._pos is None:
            self._encode_prompts()
        if not self.few_shot:
            return                      # zero-shot: reference images are unused
        paths = list(image_paths)
        if len(paths) > self.max_train_images:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(paths), self.max_train_images, replace=False)
            paths = [paths[i] for i in sorted(idx)]
        print(f"  [{self.name}] encoding {len(paths)} reference images …")
        self.reference, _ = self._window_features(paths, desc=f"reference [{self.name}]")

    @torch.no_grad()
    def score(self, image_paths: list[Path]) -> np.ndarray:
        if self._pos is None:
            self._encode_prompts()
        ref = torch.tensor(self.reference) if self.reference is not None else None
        scores = []
        for i in tqdm(range(0, len(image_paths), self.batch_size), desc=f"scoring [{self.name}]"):
            chunk = image_paths[i : i + self.batch_size]
            f, W = self._window_features(chunk, desc="")
            t = torch.tensor(f).to(self.device)
            # Language score per window: softmax over paired positive/negative
            # prompt similarity, then the strongest prompt pair.
            sp = t @ self._pos.T                       # (N*W, K)
            sn = t @ self._neg.T
            lang = torch.softmax(torch.stack([sp, sn], -1) * self.temperature,
                                 -1)[..., 0].max(-1).values
            s = lang.reshape(len(chunk), W).cpu()
            if ref is not None:
                # Few-shot term: distance to the nearest normal reference window.
                # Both feature sets are L2-normalised, so the Euclidean distance
                # lies in [0, 2] and d/2 is on the same [0, 1] scale as the
                # language probability -- the two can then be averaged without
                # one silently dominating.
                d = torch.cdist(t.cpu(), ref).min(dim=1).values.reshape(len(chunk), W)
                s = 0.5 * s + 0.5 * (d / 2.0).clamp(0, 1)
            scores.append(s.max(dim=1).values.numpy())
        return np.concatenate(scores)
