from __future__ import annotations
"""Distribution divergence measures and a feature-space visualisation.

Mean pairwise cosine distance plus a linear-separability check establishes that
the domains are distinct, but neither has an accepted interpretation as a
distribution-level statistic. This computes the standard apparatus instead --
t-SNE, MMD and Frechet distance -- so that the separation is demonstrated with
measures that have established meanings rather than one ad hoc statistic.

Three quantities per dataset pair, all on frozen DINOv2 CLS embeddings:

  MMD^2       maximum mean discrepancy with an RBF kernel, unbiased estimator.
              Zero iff the two distributions coincide; a kernel two-sample test
              statistic, so it says whether the samples could share a
              distribution at all.
  Frechet     the FID construction: ||mu1-mu2||^2 + Tr(C1 + C2 - 2(C1 C2)^1/2).
              Sensitive to differences in covariance as well as in mean, which
              mean pairwise distance is not.
  cosine      mean pairwise cosine distance, retained so the kernel- and
              moment-based measures can be checked against it.

The kernel bandwidth uses the median heuristic on the pooled sample, which
avoids choosing a scale that flatters the result.

Also writes a t-SNE projection of all three domains for the figure.

Usage:  python analyze_divergence.py [--n 200] [--seeds 0 1 2 3 4]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

import config as cfg
from analyze_domain_gap import NAMES, sample_paths, embed, _unit


def mmd2_rbf(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Unbiased MMD^2. Diagonal terms excluded so E[MMD^2]=0 under H0."""
    def k(a, b):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.exp(-gamma * d)
    m, n = len(x), len(y)
    kxx, kyy, kxy = k(x, x), k(y, y), k(x, y)
    np.fill_diagonal(kxx, 0.0)
    np.fill_diagonal(kyy, 0.0)
    return float(kxx.sum() / (m * (m - 1)) + kyy.sum() / (n * (n - 1))
                 - 2 * kxy.mean())


def frechet(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    mu1, mu2 = x.mean(0), y.mean(0)
    c1, c2 = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    covmean = sqrtm(c1 @ c2)
    if np.iscomplexobj(covmean):        # numerical noise in the matrix sqrt
        covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))


def median_gamma(z: np.ndarray) -> float:
    """Median heuristic: gamma = 1 / (2 * median pairwise squared distance)."""
    idx = np.random.default_rng(0).choice(len(z), min(len(z), 500), replace=False)
    s = z[idx]
    d = ((s[:, None, :] - s[None, :, :]) ** 2).sum(-1)
    med = np.median(d[d > 0])
    return float(1.0 / (2 * med)) if med > 0 else 1.0


def main(n: int, seeds: list[int]) -> None:
    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(cfg.DINOV2_MODELS["base"])
    model = AutoModel.from_pretrained(cfg.DINOV2_MODELS["base"]).to(device).eval()

    per_pair = {p: {"mmd2": [], "frechet": [], "cosine": []}
                for p in combinations(NAMES, 2)}
    last = None
    for s in seeds:
        E = {nm: embed(sample_paths(nm, n, s), proc, model, device) for nm in NAMES}
        U = {nm: _unit(E[nm]) for nm in NAMES}
        gamma = median_gamma(np.concatenate([U[nm] for nm in NAMES]))
        for a, b in combinations(NAMES, 2):
            per_pair[(a, b)]["mmd2"].append(mmd2_rbf(U[a], U[b], gamma))
            per_pair[(a, b)]["frechet"].append(frechet(E[a], E[b]))
            per_pair[(a, b)]["cosine"].append(float((1 - U[a] @ U[b].T).mean()))
        last = U

    print(f"  {'pair':22s} {'MMD^2':>18s} {'Frechet':>16s} {'cosine':>16s}")
    out = {}
    for (a, b), v in per_pair.items():
        def ms(k):
            arr = np.array(v[k]); return f"{arr.mean():.4f}+/-{arr.std(ddof=1):.4f}"
        print(f"  {a+' vs '+b:22s} {ms('mmd2'):>18s} {ms('frechet'):>16s} {ms('cosine'):>16s}")
        out[f"{a}__{b}"] = {k: [float(x) for x in v[k]] for k in v}

    # t-SNE on the last seed's embeddings, for the figure.
    from sklearn.manifold import TSNE
    X = np.concatenate([last[nm] for nm in NAMES])
    lab = np.concatenate([[i] * len(last[nm]) for i, nm in enumerate(NAMES)])
    emb = TSNE(n_components=2, perplexity=30, init="pca",
               random_state=0).fit_transform(X)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    for i, nm in enumerate(NAMES):
        m = lab == i
        ax.scatter(emb[m, 0], emb[m, 1], s=9, alpha=0.75, label=nm)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("DINOv2 CLS embeddings of defect-free images (t-SNE)")
    ax.legend(frameon=False, loc="best", fontsize=9)
    fig.tight_layout()
    figs = cfg.RESULTS_DIR / "figures"; figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"tsne_domains.{ext}", dpi=200)
    print(f"  figure -> {figs/'tsne_domains.pdf'}")

    json.dump(out, open(cfg.RESULTS_DIR / "divergence.json", "w"), indent=2)
    print(f"  written to {cfg.RESULTS_DIR/'divergence.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    main(a.n, a.seeds)
