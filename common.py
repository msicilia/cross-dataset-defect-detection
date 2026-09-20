"""Helpers shared by the experiment drivers: device choice, dataset
construction, reference sampling, test-set capping and result files."""
from __future__ import annotations

import contextlib
import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np

import config as cfg
from datasets import MVTecDataset, SDNETDataset, VISIONDataset


def pick_device(requested: str | None = None) -> str:
    """The requested device, else $DEVICE, else cuda, mps, cpu in that order."""
    import torch

    requested = requested or os.environ.get("DEVICE")
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_dataset(name: str):
    root = cfg.DATASET_PATHS[name]
    if name == "mvtec":
        return MVTecDataset(root, categories=cfg.MVTEC_CATEGORIES)
    if name == "sdnet":
        return SDNETDataset(root)
    if name == "vision":
        return VISIONDataset(root, categories=cfg.VISION_CATEGORIES)
    raise ValueError(f"unknown dataset: {name}")


def seeded_rng(seed: int, tag: int) -> np.random.Generator:
    """Independent stream per (seed, purpose)."""
    return np.random.default_rng([seed, tag])


def random_subset(paths: list, n: int, rng: np.random.Generator) -> list:
    """Up to n paths in random order; callers take prefixes as nested subsets,
    and a random order makes every prefix mix categories."""
    idx = rng.permutation(len(paths))[:n]
    return [paths[i] for i in idx]


def stratified_test(split, max_test: int | None, seed: int):
    """Label-stratified subsample of a test split, in split order.

    Splits of at most max_test images are returned whole. Otherwise each class
    keeps its share of max_test, rounded, and at least one image; when the
    rounded counts exceed max_test, the larger class gives up the excess, so
    the result has at most max(max_test, 2) images.
    """
    paths = list(split.image_paths)
    labels = np.array(split.labels)
    if max_test is None or len(paths) <= max_test:
        return paths, labels
    rng = np.random.default_rng([seed, 7777])
    idx = np.arange(len(paths))
    pos, neg = idx[labels == 1], idx[labels == 0]
    frac = max_test / len(paths)
    n_pos = min(max(1, int(round(len(pos) * frac))), len(pos))
    n_neg = min(max(1, int(round(len(neg) * frac))), len(neg))
    excess = n_pos + n_neg - max_test
    if excess > 0:
        if n_pos >= n_neg:
            n_pos = max(1, n_pos - excess)
        else:
            n_neg = max(1, n_neg - excess)
    keep = np.concatenate([
        rng.choice(pos, n_pos, replace=False),
        rng.choice(neg, n_neg, replace=False),
    ])
    keep.sort()
    return [paths[i] for i in keep], labels[keep]


# ── result files ──────────────────────────────────────────────────────────────
#
# A cell directory holds scores.npz (per-image scores, labels, paths) and
# result.json (metrics, configuration, software environment). result.json is
# written last, so its presence marks a complete cell. Both are written to a
# temporary file and renamed, so an interrupted run never leaves a partial file.

def _atomic_write(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as f:
            write(f)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def environment(device: str) -> dict:
    import sklearn
    import torch
    import torchvision
    import transformers

    return {
        "device": device,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def _normalised(config: dict) -> dict:
    return json.loads(json.dumps(config))


def save_cell(out_dir: Path, config: dict, metrics: dict, device: str,
              scores=None, labels=None, paths=None) -> None:
    """Write a cell: scores.npz first (if given), then result.json."""
    if scores is not None:
        arrays = {
            "scores": np.asarray(scores, dtype=np.float64),
            "labels": np.asarray(labels, dtype=np.int8),
            "paths": np.array([str(p) for p in paths]),
        }
        _atomic_write(out_dir / "scores.npz",
                      lambda f: np.savez_compressed(f, **arrays))
    record = {**metrics, "config": _normalised(config),
              "environment": environment(device)}
    _atomic_write(out_dir / "result.json",
                  lambda f: f.write(json.dumps(record, indent=2).encode()))


def cell_done(out_dir: Path, config: dict, needs_scores: bool = True) -> bool:
    """True if the cell is complete and was produced with this configuration.

    A complete cell produced with a different configuration raises instead of
    being reused or overwritten.
    """
    result = out_dir / "result.json"
    if not result.exists():
        return False
    if needs_scores and not (out_dir / "scores.npz").exists():
        return False
    stored = json.loads(result.read_text()).get("config")
    if stored != _normalised(config):
        raise RuntimeError(
            f"{out_dir} was produced with a different configuration.\n"
            f"  stored:   {stored}\n  expected: {_normalised(config)}\n"
            f"Move the directory aside to recompute it.")
    return True
