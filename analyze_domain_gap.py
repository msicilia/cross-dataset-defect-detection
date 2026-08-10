from __future__ import annotations
"""Quantify the domain gap between the three datasets in DINOv2 feature space.

Embeds defect-free images from each dataset with the frozen DINOv2 ViT-B/14
backbone (the same encoder used by DINO-PatchCore), then, averaged over several
random samples (seeds):
  * computes the MEAN PAIRWISE cosine distance between images of each dataset pair
  * checks linear separability of each pair (5-fold CV accuracy)

We use mean-pairwise distance rather than the distance between cluster *centroids*:
MVTec and VISION are each diffuse (multi-category) clusters whose centroids can
coincide even though their images are far apart, so centroid distance is unstable
and can mis-order the pairs. Mean-pairwise distance is stable across seeds and
faithful to how far typical images sit apart. We deliberately do not show a 2-D
projection: PCA captures only ~33% of the variance here (mis-showing MVTec/VISION
as overlapping) and t-SNE distances are non-metric, so neither faithfully conveys
the distance ordering.

Output:
    results/figures/domain_gap.pdf   (mean-pairwise distance matrix)
    results/figures/domain_gap.png   (for the plain-language docs)

Usage:
    python analyze_domain_gap.py [--n 120] [--seeds 0 1 2 3 4]
"""
import argparse
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import config as cfg

NAMES   = ["SDNET", "MVTec", "VISION"]
COLORS  = {"SDNET": "#1f77b4", "MVTec": "#ff7f0e", "VISION": "#2ca02c"}
FIGURES = cfg.RESULTS_DIR / "figures"


def _imgs(directory: Path, pattern: str) -> list[Path]:
    return [p for p in sorted(Path(directory).glob(pattern)) if not p.name.startswith("._")]


def sample_paths(name: str, n: int, seed: int) -> list[Path]:
    data = cfg.DATA_ROOT
    if name == "SDNET":
        ps = _imgs(data / "sdnet2018" / "W" / "UW", "*.jpg")
    elif name == "MVTec":
        ps = []
        for c in cfg.MVTEC_CATEGORIES:
            ps += _imgs(data / "mvtec_anomaly_detection" / c / "train" / "good", "*.png")
    else:  # VISION
        ps = []
        for c in cfg.VISION_CATEGORIES:
            ps += _imgs(data / "vision_dataset" / c / "inference", "*.jpg")
    random.Random(seed).shuffle(ps)
    return ps[:n]


@torch.no_grad()
def embed(paths: list[Path], proc, model, device: str) -> np.ndarray:
    out = []
    for i in range(0, len(paths), 16):
        ims = [Image.open(p).convert("RGB").resize((256, 256)) for p in paths[i:i + 16]]
        inp = proc(images=ims, return_tensors="pt").to(device)
        cls = model(**inp).last_hidden_state[:, 0, :]   # CLS token
        out.append(cls.cpu().numpy())
    return np.concatenate(out)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main(n: int, seeds: list[int]) -> None:
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    proc  = AutoImageProcessor.from_pretrained(cfg.DINOV2_MODELS["base"])
    model = AutoModel.from_pretrained(cfg.DINOV2_MODELS["base"]).to(device).eval()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    pair_dist = {p: [] for p in combinations(NAMES, 2)}   # mean-pairwise per seed
    pair_sep  = {p: [] for p in combinations(NAMES, 2)}   # separability per seed
    for s in seeds:
        U = {nm: _unit(embed(sample_paths(nm, n, s), proc, model, device)) for nm in NAMES}
        for a, b in combinations(NAMES, 2):
            pair_dist[(a, b)].append(float((1 - U[a] @ U[b].T).mean()))
            X = np.vstack([U[a], U[b]]); y = np.array([0] * len(U[a]) + [1] * len(U[b]))
            pair_sep[(a, b)].append(cross_val_score(
                LogisticRegression(max_iter=1000), X, y, cv=5).mean())

    # mean +/- std matrices
    M = np.full((3, 3), np.nan); S = np.zeros((3, 3))
    for a, b in combinations(NAMES, 2):
        i, j = NAMES.index(a), NAMES.index(b)
        m, sd = np.mean(pair_dist[(a, b)]), np.std(pair_dist[(a, b)])
        M[i, j] = M[j, i] = m; S[i, j] = S[j, i] = sd

    print(f"Mean pairwise cosine distance over {len(seeds)} seeds (mean +/- std):")
    for a, b in combinations(NAMES, 2):
        i, j = NAMES.index(a), NAMES.index(b)
        sep = np.mean(pair_sep[(a, b)])
        print(f"  {a}-{b:7}: {M[i, j]:.3f} +/- {S[i, j]:.3f}   separability={sep:.3f}")

    # ── figure: mean-pairwise distance matrix ───────────────────────────────────
    # A 2-D projection is deliberately NOT shown: PCA captures only ~33% of the
    # variance (mis-showing MVTec/VISION as overlapping) and t-SNE distances are
    # not metric; only this matrix faithfully carries the "VISION is farthest"
    # claim. Separability is reported in the text instead.
    off = M[~np.isnan(M)]
    masked = np.ma.masked_invalid(M)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("#dddddd")

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(masked, cmap=cmap, vmin=off.min() - 0.005, vmax=off.max() + 0.005)
    ax.set_xticks(range(3)); ax.set_xticklabels(NAMES, fontsize=10)
    ax.set_yticks(range(3)); ax.set_yticklabels(NAMES, fontsize=10)
    mid = (off.min() + off.max()) / 2
    for i in range(3):
        for j in range(3):
            if np.isnan(M[i, j]):
                ax.text(j, i, "--", ha="center", va="center", color="#888", fontsize=11)
            else:
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white" if M[i, j] < mid else "black", fontsize=12)
    ax.set_title("Mean pairwise distance (DINOv2 features)", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "domain_gap.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "domain_gap.png", dpi=140, bbox_inches="tight")
    print(f"\n-> wrote {FIGURES/'domain_gap.pdf'} and .png  (n={n}/dataset, seeds={seeds})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="images sampled per dataset")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    main(a.n, a.seeds)
