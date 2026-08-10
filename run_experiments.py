from __future__ import annotations
"""Main experiment runner: cross-dataset generalisation benchmark.

For each (method, source_dataset, target_dataset) triple, fits the method on
the source normal training split and evaluates on the target test split.
Results are saved incrementally to results/raw/<method>/<src>__<tgt>.json
so that a crash does not lose progress.

Usage:
    python run_experiments.py [--methods M [M ...]] [--datasets D [D ...]]
                              [--seeds S [S ...]] [--device cuda|cpu]
                              [--skip-existing]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

import config as cfg
from datasets import MVTecDataset, SDNETDataset, VISIONDataset
from methods import DINOPatchCore, PatchCore, SPADE, PaDiM, WinCLIP, CLIPZS
from evaluate import image_level_metrics, pixel_level_metrics, EvalResult


# ── dataset factory ───────────────────────────────────────────────────────────

def build_dataset(name: str, seed: int = 0):
    root = cfg.DATASET_PATHS[name]
    if name == "mvtec":
        return MVTecDataset(root, categories=cfg.MVTEC_CATEGORIES)
    if name == "sdnet":
        return SDNETDataset(root, seed=seed)  # all subsets (W/D/P) when manifest present
    if name == "vision":
        return VISIONDataset(root, categories=cfg.VISION_CATEGORIES)
    raise ValueError(f"Unknown dataset: {name}")


# ── method factory ────────────────────────────────────────────────────────────

def build_method(name: str, device: str):
    if name == "dino_patchcore_small":
        return DINOPatchCore(backbone=cfg.DINOV2_MODELS["small"], device=device,
                             **{k: v for k, v in cfg.PATCHCORE.items()
                                if k != "image_size"}, image_size=cfg.PATCHCORE["image_size"])
    if name == "dino_patchcore_base":
        return DINOPatchCore(backbone=cfg.DINOV2_MODELS["base"], device=device,
                             **{k: v for k, v in cfg.PATCHCORE.items()
                                if k != "image_size"}, image_size=cfg.PATCHCORE["image_size"])
    if name == "dino_patchcore_large":
        return DINOPatchCore(backbone=cfg.DINOV2_MODELS["large"], device=device,
                             **{k: v for k, v in cfg.PATCHCORE.items()
                                if k != "image_size"}, image_size=cfg.PATCHCORE["image_size"])
    if name == "patchcore":
        return PatchCore(device=device,
                         coreset_ratio=cfg.PATCHCORE["coreset_ratio"],
                         max_train_images=cfg.PATCHCORE["max_train_images"],
                         batch_size=cfg.PATCHCORE["batch_size"])
    if name == "spade":
        return SPADE(device=device,
                     k=cfg.SPADE["k"],
                     max_train_images=cfg.PATCHCORE["max_train_images"],
                     batch_size=cfg.PATCHCORE["batch_size"])
    if name == "padim":
        return PaDiM(device=device,
                     n_dims=cfg.PADIM["n_dims"],
                     max_train_images=cfg.PATCHCORE["max_train_images"],
                     batch_size=cfg.PATCHCORE["batch_size"])
    if name in ("winclip", "winclip_plus"):
        return WinCLIP(model_id=cfg.CLIP_MODEL, device=device,
                       prompts=cfg.CLIP_ZS["prompts"],
                       image_size=cfg.CLIP_ZS["image_size"],
                       few_shot=(name == "winclip_plus"),
                       max_train_images=cfg.PATCHCORE["max_train_images"],
                       batch_size=cfg.WINCLIP["batch_size"])
    if name == "clip_zs":
        return CLIPZS(model_id=cfg.CLIP_MODEL, device=device,
                      prompts=cfg.CLIP_ZS["prompts"],
                      image_size=cfg.CLIP_ZS["image_size"],
                      batch_size=cfg.CLIP_ZS["batch_size"])
    raise ValueError(f"Unknown method: {name}")


# ── result serialisation ──────────────────────────────────────────────────────

def save_result(out_dir: Path, result: EvalResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "result.json", "w") as f:
        json.dump({
            "image_auroc":      result.image_auroc,
            "image_ap":         result.image_ap,
            "pixel_auroc":      result.pixel_auroc,
            "pixel_ap":         result.pixel_ap,
            "threshold":        result.threshold,
            "f1_at_threshold":  result.f1_at_threshold,
        }, f, indent=2)


def save_scores(out_dir: Path, scores, labels, paths) -> None:
    """Persist the per-image scores behind an aggregate metric.

    Storing only summary metrics would mean that any per-category breakdown,
    bootstrap confidence interval or score-distribution analysis required
    re-running the model. Keeping the raw vectors makes those pure post-hoc
    analyses. Paths are stored relative to nothing in particular --
    they are kept verbatim so that the dataset category can be recovered from
    the directory structure (e.g. MVTec's <category>/test/<defect>/img.png).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "scores.npz",
        scores=np.asarray(scores, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int8),
        paths=np.array([str(p) for p in paths]),
    )


def _cell_done(out_dir: Path) -> bool:
    """A cell counts as complete only with both its metrics and its raw scores."""
    return (out_dir / "result.json").exists() and (out_dir / "scores.npz").exists()


def load_result(out_dir: Path) -> EvalResult | None:
    p = out_dir / "result.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    return EvalResult(**d)


# ── main loop ─────────────────────────────────────────────────────────────────

def run(
    methods: list[str],
    datasets: list[str],
    seeds: list[int],
    device: str,
    skip_existing: bool,
    compute_pixel: bool,
) -> None:
    results_root = cfg.RESULTS_DIR / "raw"

    for method_name in methods:
        print(f"\n{'='*60}")
        print(f"METHOD: {method_name}")
        print(f"{'='*60}")

        # Reference-free methods do not depend on the source dataset at all, so
        # they are fitted once and evaluated on each target. Running them per
        # (source, seed) like the others would recompute identical numbers 15
        # times. WinCLIP+ is NOT in this set: its few-shot variant does use
        # reference images, which is precisely the contrast being tested.
        clip_zs_mode = method_name in ("clip_zs", "winclip")

        method = build_method(method_name, device)

        if clip_zs_mode:
            method.fit([])  # no-op

        for src_name in datasets:
            if not clip_zs_mode:
                # Re-instantiate for each seed-average (or once if no seed variation)
                for seed in seeds:
                    # Fitting is the expensive half, so skip it outright when
                    # every target for this (source, seed) is already stored --
                    # otherwise a resumed run rebuilds banks it never uses.
                    if skip_existing and all(
                        _cell_done(results_root / method_name / f"seed{seed}"
                                   / f"{src_name}__{t}")
                        for t in datasets
                    ):
                        print(f"\n  [{src_name} (seed={seed})] all targets done, "
                              f"skipping fit")
                        continue
                    src_ds = build_dataset(src_name, seed)
                    method = build_method(method_name, device)
                    train_split = src_ds.normal_train()
                    print(f"\n  Fitting on {src_name} (seed={seed}) …")
                    method.fit(train_split.image_paths, seed=seed)
                    _evaluate_all_targets(
                        method, method_name, src_name, datasets, seed,
                        results_root, skip_existing, compute_pixel
                    )
            else:
                # CLIP-ZS: single evaluation, no source dependency
                _evaluate_all_targets(
                    method, method_name, "none", datasets, 0,
                    results_root, skip_existing, compute_pixel
                )
                break  # only one "source" needed for CLIP-ZS


def _evaluate_all_targets(
    method,
    method_name: str,
    src_name: str,
    datasets: list[str],
    seed: int,
    results_root: Path,
    skip_existing: bool,
    compute_pixel: bool,
) -> None:
    for tgt_name in datasets:
        out_dir = results_root / method_name / f"seed{seed}" / f"{src_name}__{tgt_name}"
        # A cell counts as done only if the per-image scores are present too,
        # not just the summary metrics.
        if skip_existing and _cell_done(out_dir):
            print(f"    [{src_name} → {tgt_name}] already done, skipping")
            continue

        print(f"    Evaluating: {src_name} → {tgt_name} …", end=" ", flush=True)
        tgt_ds = build_dataset(tgt_name, seed)
        test   = tgt_ds.test()

        scores = method.score(test.image_paths)
        labels = np.array(test.labels)
        result = image_level_metrics(scores, labels)

        if compute_pixel:
            try:
                smaps = method.score_maps(test.image_paths)
                p_auroc, p_ap = pixel_level_metrics(smaps, test.mask_paths)
                result.pixel_auroc = p_auroc
                result.pixel_ap    = p_ap
            except NotImplementedError:
                pass

        save_result(out_dir, result)
        save_scores(out_dir, scores, labels, test.image_paths)
        print(f"AUROC={result.image_auroc:.4f}  AP={result.image_ap:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods",  nargs="+",
                   default=["dino_patchcore_base", "patchcore", "clip_zs"])
    p.add_argument("--datasets", nargs="+", default=cfg.DATASET_NAMES)
    p.add_argument("--seeds",    nargs="+", type=int, default=[0])
    p.add_argument("--device",   default="cuda" if __import__("torch").cuda.is_available() else "mps" if __import__("torch").backends.mps.is_available() else "cpu")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--no-pixel",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        methods=args.methods,
        datasets=args.datasets,
        seeds=args.seeds,
        device=args.device,
        skip_existing=args.skip_existing,
        compute_pixel=not args.no_pixel,
    )
