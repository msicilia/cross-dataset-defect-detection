from __future__ import annotations
"""Texture-vs-object structure: analysis for the MVTec-split study.

Produces a two-panel figure over four groups
    SDNET | MVTec-tex (tile,wood,grid) | MVTec-obj (metal_nut,screw) | VISION
  (a) mean pairwise cosine distance in DINOv2 feature space (5 samples)
  (b) cross-group transfer AUROC for DINO-PatchCore (from run_mvtec_split.py)

Both should reveal two super-clusters: flat surfaces {SDNET, MVTec-tex} and
metallic objects {MVTec-obj, VISION}.

Output:
    results/figures/mvtec_split.pdf / .png
    results/mvtec_split.json

Usage:
    python analyze_mvtec_split.py [--n 90] [--seeds 0 1 2 3 4]
"""
import argparse
import json
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

GROUPS = ["sdnet", "mvtec_tex", "mvtec_obj", "vision"]
LABEL  = {"sdnet": "SDNET", "mvtec_tex": "MVTec-tex", "mvtec_obj": "MVTec-metal", "vision": "VISION"}
MVTEC_TEX = ["tile", "wood", "grid"]
MVTEC_OBJ = ["metal_nut", "screw"]
FIG = cfg.RESULTS_DIR / "figures"
RAW = cfg.RESULTS_DIR / "raw" / "dino_patchcore_mvtecsplit"


def _imgs(d, p): return [x for x in sorted(Path(d).glob(p)) if not x.name.startswith("._")]


def pool(g):
    data = cfg.DATA_ROOT
    if g == "sdnet":
        return _imgs(data / "sdnet2018" / "W" / "UW", "*.jpg")
    if g == "vision":
        ps = []
        for c in cfg.VISION_CATEGORIES:
            ps += _imgs(data / "vision_dataset" / c / "inference", "*.jpg")
        return ps
    cats = MVTEC_TEX if g == "mvtec_tex" else MVTEC_OBJ
    ps = []
    for c in cats:
        ps += _imgs(data / "mvtec_anomaly_detection" / c / "train" / "good", "*.png")
    return ps


@torch.no_grad()
def embed(paths, proc, model, device):
    out = []
    for i in range(0, len(paths), 16):
        ims = [Image.open(p).convert("RGB").resize((256, 256)) for p in paths[i:i + 16]]
        out.append(model(**proc(images=ims, return_tensors="pt").to(device))
                   .last_hidden_state[:, 0, :].cpu().numpy())
    return np.concatenate(out)


def feature_matrix(n, seeds):
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(cfg.DINOV2_MODELS["base"])
    model = AutoModel.from_pretrained(cfg.DINOV2_MODELS["base"]).to(device).eval()
    POOL = {g: pool(g) for g in GROUPS}
    dist = {p: [] for p in combinations(GROUPS, 2)}
    for s in seeds:
        rng = random.Random(s); U = {}
        for g in GROUPS:
            ps = POOL[g][:]; rng.shuffle(ps)
            e = embed(ps[:n], proc, model, device)
            U[g] = e / np.linalg.norm(e, axis=1, keepdims=True)
        for a, b in combinations(GROUPS, 2):
            dist[(a, b)].append(float((1 - U[a] @ U[b].T).mean()))
    M = np.full((4, 4), np.nan)
    for a, b in combinations(GROUPS, 2):
        i, j = GROUPS.index(a), GROUPS.index(b)
        M[i, j] = M[j, i] = np.mean(dist[(a, b)])
    return M


def transfer_matrix():
    M = np.full((4, 4), np.nan)
    for i, s in enumerate(GROUPS):
        for j, t in enumerate(GROUPS):
            vals = [json.load(open(f))["image_auroc"]
                    for f in sorted(RAW.glob(f"{s}__{t}/seed*/result.json"))]
            if vals:
                M[i, j] = np.mean(vals)
    return M


def _heat(ax, M, title, cmap, vlo, vhi, fmt, mask_diag=False, lo_text_thresh=None):
    A = M.copy()
    if mask_diag:
        np.fill_diagonal(A, np.nan)
    masked = np.ma.masked_invalid(A)
    cm = matplotlib.colormaps[cmap].copy(); cm.set_bad("#dddddd")
    im = ax.imshow(masked, cmap=cm, vmin=vlo, vmax=vhi)
    ax.set_xticks(range(4)); ax.set_xticklabels([LABEL[g] for g in GROUPS], fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels([LABEL[g] for g in GROUPS], fontsize=8)
    for i in range(4):
        for j in range(4):
            if np.isnan(A[i, j]):
                ax.text(j, i, "--", ha="center", va="center", color="#888", fontsize=9)
            else:
                c = "white" if (lo_text_thresh is not None and A[i, j] < lo_text_thresh) else "black"
                ax.text(j, i, fmt.format(A[i, j]), ha="center", va="center", color=c, fontsize=9)
    ax.set_title(title, fontsize=10)
    return im


def main(n, seeds):
    Mf = feature_matrix(n, seeds)
    Mt = transfer_matrix()

    print("Feature mean-pairwise distance:");  print(np.round(Mf, 3))
    print("Transfer AUROC (rows=source, cols=target):"); print(np.round(Mt, 3))
    json.dump({"groups": GROUPS,
               "feature_distance": Mf.tolist(),
               "transfer_auroc": Mt.tolist()},
              open(cfg.RESULTS_DIR / "mvtec_split.json", "w"), indent=2)

    have_t = not np.all(np.isnan(Mt))
    ncol = 2 if have_t else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.7 * ncol, 3.8))
    axes = np.atleast_1d(axes)
    off = Mf[~np.isnan(Mf)]
    im0 = _heat(axes[0], Mf, "(a) Feature distance", "viridis",
                off.min() - 0.005, off.max() + 0.005, "{:.3f}",
                mask_diag=True, lo_text_thresh=(off.min() + off.max()) / 2)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    if have_t:
        im1 = _heat(axes[1], Mt, "(b) Transfer AUROC", "RdYlGn", 0.40, 0.95, "{:.2f}",
                    lo_text_thresh=0.62)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "mvtec_split.pdf", bbox_inches="tight")
    fig.savefig(FIG / "mvtec_split.png", dpi=140, bbox_inches="tight")
    print(f"-> wrote {FIG/'mvtec_split.pdf'} (transfer panel: {'yes' if have_t else 'pending'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=90)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    main(a.n, a.seeds)
