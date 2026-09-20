"""Feature distance and transfer between four groups with MVTec AD split.

Groups (keys as in run_mvtec_split.py): sdnet (SDNET2018), mvtec_tex (MVTec tile,
wood, grid), mvtec_obj (MVTec metal_nut, screw) and vision (VISION). The key
mvtec_obj names result directories and is kept; it is labelled MVTec-metal, as
in the paper. Clusters: surface = {sdnet, mvtec_tex}, metallic = {mvtec_obj,
vision}.

Feature distance: the protocol of analyze_divergence.py, with its helpers and
defaults: n images per group per draw (default N_PER_GROUP) from the
normal_train() split of each group as built by run_mvtec_split.py, DRAWS draws,
the same per-group sampling tag, and per pair and draw the same measures,
including the RBF bandwidth from the median heuristic on that pair's pooled
sample. Every group pool must hold at least n images.

Transfer: DINO-PatchCore image AUROC from run_mvtec_split.py,
results/raw/dino_patchcore_mvtecsplit/<source>__<target>/seed<s>/result.json,
mean and SD (ddof=1) over cfg.SEEDS. Within-surface, within-metallic and
cross-cluster transfer are the means of the seed-mean off-diagonal cells whose
source and target lie in the same surface cluster, the same metallic cluster,
or different clusters.

Figure: (a) mean pairwise cosine distance, (b) transfer AUROC; the colour
range of (b) is symmetric about chance (0.5) and spans the data.

Usage:  python analyze_mvtec_split.py [--n 200] [--draws 5] [--out results] [--device cpu]
Output: <out>/mvtec_split.json, <out>/figures/mvtec_split.{pdf,png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import config as cfg
from analyze_divergence import (DRAWS, MEASURES, MODEL, N_PER_GROUP, pairwise_measures,
                                sample_embeddings, summarise, write_json)
from common import pick_device
from run_mvtec_split import GROUPS, MVTEC_GROUPS, build_group

LABEL = {"sdnet": "SDNET", "mvtec_tex": "MVTec-tex", "mvtec_obj": "MVTec-metal",
         "vision": "VISION"}
CLUSTERS = {"surface": ["sdnet", "mvtec_tex"], "metallic": ["mvtec_obj", "vision"]}
RAW = cfg.RESULTS_DIR / "raw" / "dino_patchcore_mvtecsplit"


def group_pools() -> dict[str, list[Path]]:
    return {g: list(build_group(g).normal_train().image_paths) for g in GROUPS}


def transfer() -> dict:
    expected = {(s, t, seed): RAW / f"{s}__{t}" / f"seed{seed}" / "result.json"
                for s in GROUPS for t in GROUPS for seed in cfg.SEEDS}
    missing = [str(p) for p in expected.values() if not p.exists()]
    if missing:
        raise SystemExit("missing MVTec-split runs:\n  " + "\n  ".join(missing))
    per_seed = np.array([[[json.loads(expected[(s, t, seed)].read_text())["image_auroc"]
                           for seed in cfg.SEEDS] for t in GROUPS] for s in GROUPS])
    mean = per_seed.mean(axis=2)
    sd = per_seed.std(axis=2, ddof=1)

    cluster_of = {g: c for c, gs in CLUSTERS.items() for g in gs}
    buckets = {"within_surface": [], "within_metallic": [], "cross_cluster": []}
    for i, s in enumerate(GROUPS):
        for j, t in enumerate(GROUPS):
            if i == j:
                continue
            if cluster_of[s] != cluster_of[t]:
                key = "cross_cluster"
            else:
                key = f"within_{cluster_of[s]}"
            buckets[key].append((s, t))
    averages = {k: {"mean": float(np.mean([mean[GROUPS.index(s), GROUPS.index(t)]
                                           for s, t in cells])),
                    "cells": [f"{s}__{t}" for s, t in cells]}
                for k, cells in buckets.items()}
    return {"per_seed": per_seed, "mean": mean, "sd": sd, "averages": averages}


def heatmap(ax, M, title, cmap_name, vmin, vmax, fmt):
    import matplotlib

    A = np.ma.masked_invalid(M)
    cmap = matplotlib.colormaps[cmap_name].copy()
    cmap.set_bad("#dddddd")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(A, cmap=cmap, norm=norm)
    ticks = [LABEL[g] for g in GROUPS]
    ax.set_xticks(range(len(GROUPS)), ticks, fontsize=8, rotation=30, ha="right")
    ax.set_yticks(range(len(GROUPS)), ticks, fontsize=8)
    for i in range(len(GROUPS)):
        for j in range(len(GROUPS)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "--", ha="center", va="center", color="#555555", fontsize=9)
                continue
            r, g, b, _ = cmap(norm(M[i, j]))
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=9,
                    color="black" if luminance > 0.5 else "white")
    ax.set_title(title, fontsize=10)
    return im


def figure(distance: np.ndarray, transfer_mean: np.ndarray, out_dir: Path, draws: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    # TrueType, not matplotlib's default Type 3: Type 3 renders poorly at
    # the sizes these figures are printed at, and IEEE rejects Type 3.
    matplotlib.rcParams["pdf.fonttype"] = 42
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    off = distance[~np.isnan(distance)]
    im0 = heatmap(axes[0], distance, f"(a) Mean pairwise cosine distance ({draws} draws)",
                  "viridis", off.min(), off.max(), "{:.3f}")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    half = float(np.abs(transfer_mean - 0.5).max())
    im1 = heatmap(axes[1], transfer_mean,
                  f"(b) Transfer AUROC (mean over {len(cfg.SEEDS)} seeds)",
                  "RdYlGn", 0.5 - half, 0.5 + half, "{:.2f}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figs / f"mvtec_split.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure written to {figs / 'mvtec_split.pdf'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_GROUP, help="images per group per draw")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", type=Path, default=cfg.RESULTS_DIR)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    tr = transfer()
    device = pick_device(args.device)
    pools = group_pools()
    print("  pool sizes: " + ", ".join(f"{g} {len(p)}" for g, p in pools.items()))
    _, emb = sample_embeddings(pools, args.n, args.draws, device)
    per_pair, gammas, rel_imag = pairwise_measures(emb, args.draws)
    pairs = {p: {k: summarise(v[k]) for k in MEASURES} for p, v in per_pair.items()}

    k = len(GROUPS)
    distance = {m: np.full((k, k), np.nan) for m in MEASURES}
    for i, a in enumerate(GROUPS):
        for j, b in enumerate(GROUPS):
            if i < j:
                for m in MEASURES:
                    distance[m][i, j] = distance[m][j, i] = pairs[f"{a}__{b}"][m]["mean"]

    np.set_printoptions(precision=3, suppress=True)
    print(f"\n  groups: {GROUPS}")
    print(f"  mean pairwise cosine distance (n={args.n}, {args.draws} draws):")
    print(distance["cosine"])
    print("  transfer AUROC mean (rows source, columns target):")
    print(tr["mean"])
    print("  transfer AUROC SD (ddof=1):")
    print(tr["sd"])
    for name, v in tr["averages"].items():
        print(f"  {name:16s} {v['mean']:.3f}  over {v['cells']}")

    figure(distance["cosine"], tr["mean"], args.out, args.draws)

    def nan_to_none(M):
        return [[None if np.isnan(x) else float(x) for x in row] for row in M]

    write_json(args.out / "mvtec_split.json", {
        "config": {"model": MODEL, "n_per_group": args.n, "draws": list(range(args.draws)),
                   "seeds": list(cfg.SEEDS), "groups": GROUPS, "labels": LABEL,
                   "mvtec_groups": MVTEC_GROUPS, "clusters": CLUSTERS, "device": device},
        "groups": GROUPS,
        "pool_sizes": {g: len(p) for g, p in pools.items()},
        "feature_distance": nan_to_none(distance["cosine"]),
        "feature_pairs": pairs,
        "rbf_gamma_per_pair_per_draw": gammas,
        "frechet_max_rel_imag_per_draw": rel_imag,
        "transfer_auroc": tr["mean"].tolist(),
        "transfer_auroc_sd": tr["sd"].tolist(),
        "transfer_auroc_per_seed": {f"{s}__{t}": tr["per_seed"][i, j].tolist()
                                    for i, s in enumerate(GROUPS) for j, t in enumerate(GROUPS)},
        "transfer_averages": tr["averages"],
    })
    print(f"  written to {args.out / 'mvtec_split.json'}")


if __name__ == "__main__":
    main()
