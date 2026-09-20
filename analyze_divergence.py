"""Distributional distance between the datasets in DINOv2 feature space.

Embeddings: CLS token of the frozen DINOv2 ViT-B/14 backbone for defect-free
reference images, resized as in DINO-PatchCore, then L2-normalised. Every
measure below is computed on these normalised embeddings.

Sampling: for each draw d in 0..draws-1 and each dataset, n images are drawn
without replacement from the dataset's normal_train() split (SDNET2018 over its
three subsets, MVTec AD and VISION over all configured categories) with
common.seeded_rng(d, tag), where tag = zlib.crc32(b"reference-sample/<name>") and
<name> is the dataset key (sdnet, mvtec, vision; analyze_mvtec_split.py uses the
same tag with its group keys).

Measures per dataset pair and draw:

  mmd2          unbiased MMD^2 with an RBF kernel exp(-gamma ||x-y||^2);
                gamma = 1 / (2 * median squared distance over all distinct
                pairs of the pooled sample of the two datasets in the draw),
                so the value of a pair does not depend on the other datasets.
  frechet       ||mu_x - mu_y||^2 + Tr(C_x + C_y - 2 (C_x C_y)^{1/2}), sample
                covariances (ddof=1). The trace of the matrix square root is
                taken from scipy.linalg.sqrtm; it raises if its imaginary part,
                or its disagreement with the trace computed from the symmetric
                form C_x^{1/2} C_y C_x^{1/2}, exceeds MAX_REL_ERROR of its value.
  cosine        mean cosine distance 1 - <x, y> over all cross-dataset pairs
                (no image is paired with itself).
  separability  mean accuracy of 5-fold stratified cross-validated logistic
                regression telling the two datasets apart.

Summaries are the per-draw values, their mean and SD (ddof=1). A t-SNE
projection of draw 0 (perplexity 30, seeded) is coloured by dataset and shaded
by category.

Usage:  python analyze_divergence.py [--n 200] [--draws 5] [--out results] [--device cpu]
Output: <out>/divergence.json, <out>/figures/tsne_domains.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
import os
import zlib
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import pdist

import config as cfg
from analyze_per_category import category_of
from common import build_dataset, pick_device, random_subset, seeded_rng

MODEL = cfg.DINOV2_MODELS["base"]
N_PER_GROUP = 200
DRAWS = 5
IMAGE_SIZE = cfg.PATCHCORE["image_size"]
BATCH_SIZE = 16
CV_FOLDS = 5
TSNE_PERPLEXITY = 30
MAX_REL_ERROR = 1e-4
DISPLAY = {"sdnet": "SDNET2018", "mvtec": "MVTec AD", "vision": "VISION"}
SDNET_SUBSETS = {"D": "decks", "P": "pavements", "W": "walls"}


# ── sampling and embedding ────────────────────────────────────────────────────

def normal_images(dataset: str) -> list[Path]:
    return list(build_dataset(dataset).normal_train().image_paths)


def sample_tag(name: str) -> int:
    return zlib.crc32(f"reference-sample/{name}".encode())


def draw_sample(paths: list[Path], n: int, draw: int, name: str) -> list[Path]:
    if len(paths) < n:
        raise ValueError(f"{name}: {len(paths)} images available, {n} requested")
    return random_subset(paths, n, seeded_rng(draw, sample_tag(name)))


def load_encoder(device: str):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(MODEL, use_fast=False)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    return processor, model


def embed(paths: list[Path], processor, model, device: str) -> np.ndarray:
    """L2-normalised CLS embeddings, float64, one row per path."""
    import torch
    from PIL import Image

    out = []
    with torch.no_grad():
        for i in range(0, len(paths), BATCH_SIZE):
            ims = [Image.open(p).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
                   for p in paths[i:i + BATCH_SIZE]]
            inputs = processor(images=ims, return_tensors="pt").to(device)
            out.append(model(**inputs).last_hidden_state[:, 0, :].float().cpu().numpy())
    e = np.concatenate(out).astype(np.float64)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def sample_embeddings(pools: dict[str, list[Path]], n: int, draws: int, device: str):
    """{name: [paths per draw]}, {name: [embeddings per draw]}; each distinct
    image is embedded once."""
    samples = {name: [draw_sample(p, n, d, name) for d in range(draws)]
               for name, p in pools.items()}
    unique = sorted({p for per_draw in samples.values() for s in per_draw for p in s})
    processor, model = load_encoder(device)
    print(f"  embedding {len(unique)} images with {MODEL} on {device}")
    table = dict(zip(unique, embed(unique, processor, model, device)))
    emb = {name: [np.stack([table[p] for p in s]) for s in per_draw]
           for name, per_draw in samples.items()}
    return samples, emb


# ── measures ──────────────────────────────────────────────────────────────────

def median_heuristic_gamma(z: np.ndarray) -> float:
    return float(1.0 / (2.0 * np.median(pdist(z, "sqeuclidean"))))


def mmd2_unbiased(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    def k(a, b):
        return np.exp(-gamma * np.maximum(
            (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2 * a @ b.T, 0.0))
    m, n = len(x), len(y)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1)) - 2 * kxy.mean())


def frechet_distance(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fréchet distance and the relative size of the discarded imaginary part.

    Only Tr (C_x C_y)^{1/2} enters the distance. It is taken from sqrtm, and
    checked against the eigenvalues of the symmetric PSD matrix
    C_x^{1/2} C_y C_x^{1/2}, whose square roots sum to the same trace.
    """
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    tr = complex(np.trace(sqrtm(cx @ cy)))
    rel_imag = abs(tr.imag) / abs(tr.real)
    if not rel_imag < MAX_REL_ERROR:
        raise ArithmeticError(f"Tr sqrtm(C_x C_y) has imaginary part {rel_imag:.2e} of its real part")
    w, v = np.linalg.eigh(cx)
    half = (v * np.sqrt(np.clip(w, 0.0, None))) @ v.T
    tr_sym = float(np.sqrt(np.clip(np.linalg.eigvalsh(half @ cy @ half), 0.0, None)).sum())
    if not abs(tr.real - tr_sym) < MAX_REL_ERROR * tr_sym:
        raise ArithmeticError(f"Tr sqrtm(C_x C_y) = {tr.real} disagrees with symmetric form {tr_sym}")
    mean_term = float(((x.mean(0) - y.mean(0)) ** 2).sum())
    return mean_term + float(np.trace(cx) + np.trace(cy) - 2 * tr.real), rel_imag


def mean_cosine_distance(x: np.ndarray, y: np.ndarray) -> float:
    return float((1.0 - x @ y.T).mean())


def separability(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    X = np.vstack([x, y])
    labels = np.r_[np.zeros(len(x)), np.ones(len(y))]
    folds = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    return float(cross_val_score(LogisticRegression(max_iter=1000), X, labels, cv=folds).mean())


MEASURES = ["mmd2", "frechet", "cosine", "separability"]


def pairwise_measures(emb: dict[str, list[np.ndarray]], draws: int) -> tuple[dict, dict, list]:
    """Per-pair per-draw values of every measure, the per-pair per-draw gamma
    and the per-draw largest relative imaginary part of the Fréchet square
    roots."""
    names = list(emb)
    per_pair = {f"{a}__{b}": {k: [] for k in MEASURES} for a, b in combinations(names, 2)}
    gammas = {p: [] for p in per_pair}
    rel_imag = []
    for d in range(draws):
        worst = 0.0
        for a, b in combinations(names, 2):
            x, y = emb[a][d], emb[b][d]
            gamma = median_heuristic_gamma(np.vstack([x, y]))
            gammas[f"{a}__{b}"].append(gamma)
            fd, ri = frechet_distance(x, y)
            worst = max(worst, ri)
            v = per_pair[f"{a}__{b}"]
            v["mmd2"].append(mmd2_unbiased(x, y, gamma))
            v["frechet"].append(fd)
            v["cosine"].append(mean_cosine_distance(x, y))
            v["separability"].append(separability(x, y, d))
        rel_imag.append(worst)
    return per_pair, gammas, rel_imag


def summarise(values: list[float]) -> dict:
    return {"per_draw": values, "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


# ── t-SNE figure ──────────────────────────────────────────────────────────────

def tsne_figure(emb: dict[str, np.ndarray], samples: dict[str, list[Path]], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    # TrueType, not matplotlib's default Type 3: Type 3 renders poorly at
    # the sizes these figures are printed at, and IEEE rejects Type 3.
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    X = np.vstack([emb[nm] for nm in emb])
    if len(X) <= TSNE_PERPLEXITY:
        raise ValueError(f"t-SNE needs more than {TSNE_PERPLEXITY} points, got {len(X)}")
    proj = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init="pca",
                random_state=0).fit_transform(X)

    families = {"sdnet": ("Blues", "o"), "mvtec": ("Oranges", "s"), "vision": ("Greens", "^")}
    # Drawn at the single-column width it is printed at, so nothing is scaled down.
    fig, ax = plt.subplots(figsize=(3.5, 3.4))
    start = 0
    for nm in emb:
        cats = np.array([category_of(p, nm) for p in samples[nm]])
        rows = proj[start:start + len(cats)]
        start += len(cats)
        cmap_name, marker = families[nm]
        levels = sorted(set(cats))
        shades = matplotlib.colormaps[cmap_name](np.linspace(0.45, 0.95, len(levels)))
        for c, colour in zip(levels, shades):
            m = cats == c
            label = SDNET_SUBSETS.get(c, c) if nm == "sdnet" else c
            ax.scatter(rows[m, 0], rows[m, 1], s=12, marker=marker, color=colour,
                       edgecolors="none", alpha=0.85, label=f"{DISPLAY[nm]}: {label}")
    ax.set_xticks([])
    ax.set_yticks([])
    handles, labels = ax.get_legend_handles_labels()
    # Below the axes in three columns: beside them it took a third of the width,
    # leaving the labels unreadable once scaled to one column. The plot has no
    # tick labels, so the bottom strip has to be reserved explicitly.
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.legend(handles, labels, frameon=False, fontsize=6, loc="lower center",
               ncol=3, markerscale=1.2, columnspacing=1.0, handletextpad=0.4,
               bbox_to_anchor=(0.5, 0.0))
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"tsne_domains.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure written to {figs / 'tsne_domains.pdf'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_GROUP, help="images per dataset per draw")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", type=Path, default=cfg.RESULTS_DIR)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)

    pools = {nm: normal_images(nm) for nm in cfg.DATASETS}
    samples, emb = sample_embeddings(pools, args.n, args.draws, device)
    per_pair, gammas, rel_imag = pairwise_measures(emb, args.draws)
    pairs = {p: {k: summarise(v[k]) for k in MEASURES} for p, v in per_pair.items()}

    print(f"\n  n={args.n} per dataset, {args.draws} draws (mean +/- sd)")
    print(f"  {'pair':16s} " + " ".join(f"{k:>20s}" for k in MEASURES))
    for p, v in pairs.items():
        cols = []
        for k in MEASURES:
            sd = v[k]["sd"]
            cols.append(f"{v[k]['mean']:.4f} +/- {sd:.4f}" if sd is not None
                        else f"{v[k]['mean']:.4f}")
        print(f"  {p:16s} " + " ".join(f"{c:>20s}" for c in cols))
    for m in ("mmd2", "frechet", "cosine"):
        closest = min(pairs, key=lambda p: pairs[p][m]["mean"])
        print(f"  closest pair by {m}: {closest}")

    tsne_figure({nm: emb[nm][0] for nm in cfg.DATASETS},
                {nm: samples[nm][0] for nm in cfg.DATASETS}, args.out)

    write_json(args.out / "divergence.json", {
        "config": {"model": MODEL, "image_size": IMAGE_SIZE, "n_per_dataset": args.n,
                   "draws": list(range(args.draws)), "cv_folds": CV_FOLDS,
                   "tsne_perplexity": TSNE_PERPLEXITY, "tsne_draw": 0, "device": device},
        "pool_sizes": {nm: len(p) for nm, p in pools.items()},
        "sample_categories_draw0": {nm: {c: int(sum(category_of(p, nm) == c for p in samples[nm][0]))
                                         for c in sorted({category_of(p, nm) for p in samples[nm][0]})}
                                    for nm in cfg.DATASETS},
        "rbf_gamma_per_pair_per_draw": gammas,
        "frechet_max_rel_imag_per_draw": rel_imag,
        "pairs": pairs,
    })
    print(f"  written to {args.out / 'divergence.json'}")


if __name__ == "__main__":
    main()
