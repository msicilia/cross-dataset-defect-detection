"""Storage, fit time and inference latency of each detector.

All measurements use MVTec AD. Reference subsets are nested prefixes of one
seeded random draw from its reference images, and the timed test images are a
seeded random draw from its test split. Each detector is loaded once, warmed up
with a fit on a few reference images (PaDiM: n_dims + 1) and a scoring pass, and
then timed at every reference-set size.

Recorded per detector and reference-set size:
    stored_bytes   the fitted representation kept in addition to the frozen
                   weights: memory bank (PatchCore, DINO-PatchCore), image
                   descriptors (SPADE), per-position means, inverse covariances
                   and channel indices (PaDiM), reference window features
                   (WinCLIP+), prompt embeddings (WinCLIP, CLIP-ZS, WinCLIP+)
    fit_s          wall time of fit()
    latency_ms     wall time of score() over the timed test images, per image
Reference-free detectors (WinCLIP, CLIP-ZS) are measured once with n_ref = 0.
PaDiM at n_ref <= n_dims is recorded as not applicable.

The default reference-set sizes are 100, 250 and PATCHCORE["max_train_images"],
the bank size of the benchmark and the row read by make_tables.py. Larger sizes
are rejected, since the detectors cap their reference set at that value.

The measurement configuration (dataset, seed, n_test, ref_sizes, methods,
device, each detector's settings from run_experiments.method_params, and batch
sizes) is stored under "config" in the output. If the output already records
the same configuration, the run is skipped; --force re-measures. The output is
written atomically.

Usage:
    python measure_cost.py [--n-test 200] [--ref-sizes N ...] [--methods M ...]
                           [--output results/cost.json] [--device cuda|mps|cpu]
                           [--force]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import config as cfg
from common import build_dataset, pick_device, random_subset, seeded_rng
from run_experiments import build_method, method_params

METHODS = ["patchcore", "dino_patchcore_base", "spade", "padim",
           "winclip_plus", "winclip", "clip_zs"]
REFERENCE_FREE = {"clip_zs", "winclip"}
DATASET = "mvtec"
SEED = 0
WARMUP_IMAGES = 8
MAX_REF = cfg.PATCHCORE["max_train_images"]
REF_SIZES = sorted({n for n in (100, 250) if n < MAX_REF} | {MAX_REF})
STORED_ATTRS = ("memory_bank", "gallery", "mean", "inv_cov", "dim_idx",
                "reference", "_pos", "_neg", "_pos_feats", "_neg_feats")


def _stored_bytes(method) -> int:
    total = 0
    for attr in STORED_ATTRS:
        v = getattr(method, attr, None)
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif isinstance(v, torch.Tensor):
            total += v.element_size() * v.nelement()
    return total


def _params(method) -> int:
    module = getattr(method, "backbone", None)
    if module is None:
        module = getattr(method, "model", None)
    return sum(p.numel() for p in module.parameters()) if module is not None else 0


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _timed(fn, device: str) -> float:
    _sync(device)
    t0 = time.perf_counter()
    fn()
    _sync(device)
    return time.perf_counter() - t0


def _not_applicable(name: str, n_ref: int) -> str | None:
    if name == "padim" and n_ref <= cfg.PADIM["n_dims"]:
        return (f"PaDiM needs more reference images than feature channels "
                f"(n_ref={n_ref}, n_dims={cfg.PADIM['n_dims']})")
    return None


def _warmup_refs(name: str) -> int:
    if name in REFERENCE_FREE:
        return 0
    return cfg.PADIM["n_dims"] + 1 if name == "padim" else WARMUP_IMAGES


def measure(name: str, device: str, pool: list, test: list, ref_sizes: list[int]) -> dict:
    method = build_method(name, device)
    sizes = [0] if name in REFERENCE_FREE else ref_sizes

    method.fit(pool[:_warmup_refs(name)], seed=SEED)
    method.score(test[:WARMUP_IMAGES])

    scaling = []
    for n_ref in sizes:
        reason = _not_applicable(name, n_ref)
        if reason:
            print(f"  {name:22s} n_ref={n_ref}: not applicable ({reason})")
            scaling.append({"n_ref": n_ref, "not_applicable": reason})
            continue
        t_fit = _timed(lambda: method.fit(pool[:n_ref], seed=SEED), device)
        t_score = _timed(lambda: method.score(test), device)
        row = {"n_ref": n_ref, "fit_s": round(t_fit, 2),
               "stored_bytes": _stored_bytes(method),
               "latency_ms": round(1000 * t_score / len(test), 2)}
        scaling.append(row)
        print(f"  {name:22s} n_ref={n_ref:4d} | stored {row['stored_bytes'] / 1e6:8.2f} MB"
              f" | fit {row['fit_s']:7.1f} s | {row['latency_ms']:7.2f} ms/img")
    return {"method": name, "backbone_params_M": round(_params(method) / 1e6, 1),
            "scaling": scaling}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--ref-sizes", type=int, nargs="+", default=REF_SIZES,
                    help=f"reference-set sizes, at most {MAX_REF} (default: {REF_SIZES})")
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--device", default=None,
                    help="default: $DEVICE, else cuda, mps, cpu")
    ap.add_argument("--output", type=Path, default=cfg.RESULTS_DIR / "cost.json")
    ap.add_argument("--force", action="store_true",
                    help="re-measure even if the output records the same configuration")
    a = ap.parse_args()
    device = pick_device(a.device)
    ref_sizes = sorted(set(a.ref_sizes))
    if max(ref_sizes) > MAX_REF:
        ap.error(f"--ref-sizes must not exceed PATCHCORE['max_train_images'] = {MAX_REF}")
    if MAX_REF not in ref_sizes:
        print(f"note: n_ref={MAX_REF} is not measured; make_tables.py needs it")

    config = {"dataset": DATASET, "seed": SEED, "n_test": a.n_test,
              "ref_sizes": ref_sizes, "methods": a.methods, "device": device,
              "detectors": {m: method_params(m) for m in a.methods},
              "batch_sizes": {"patchcore": cfg.PATCHCORE["batch_size"],
                              "winclip": cfg.WINCLIP["batch_size"],
                              "clip_zs": cfg.CLIP_ZS["batch_size"]}}
    if a.output.exists() and not a.force:
        stored = json.loads(a.output.read_text()).get("config")
        if stored == config:
            print(f"{a.output} already records this configuration; skipped (--force re-measures)")
            return
        print(f"{a.output} records a different configuration; re-measuring")

    ds = build_dataset(DATASET)
    # Prefixes of one permutation, so the warm-up and timed subsets are nested.
    n_pool = max(ref_sizes + [_warmup_refs(m) for m in a.methods])
    pool = random_subset(ds.normal_train().image_paths, n_pool, seeded_rng(SEED, 0))
    test = random_subset(ds.test().image_paths, a.n_test, seeded_rng(SEED, 1))
    assert len(pool) == n_pool and len(test) == a.n_test

    rows = [measure(name, device, pool, test, ref_sizes) for name in a.methods]
    record = {
        "config": config,
        "device": device,
        "torch": torch.__version__,
        "dataset": DATASET,
        "seed": SEED,
        "n_test": len(test),
        "stored": "bytes of the fitted representation beyond the frozen weights",
        "methods": rows,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.output.with_name(f".{a.output.name}.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, a.output)
    print(f"\n  written to {a.output}")


if __name__ == "__main__":
    main()
