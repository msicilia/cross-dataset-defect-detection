from __future__ import annotations
"""Storage, latency and scalability of each detector.

Memory-bank methods carry storage and inference cost that scales with the
reference set, and that cost bears directly on the study's practical value.
It matters here specifically: the paper recommends memory-bank methods when a
morphologically matched source exists, so the cost of carrying that bank into a
deployment is part of the recommendation.

Reported per method:
  - model size:   parameters held on the accelerator
  - bank size:    what has to be stored *in addition* to the backbone after
                  fitting, which is the part that scales with reference data
  - fit time:     building that representation from 500 reference images
  - latency:      per-image inference, measured after a warm-up pass
  - scaling:      bank size and latency at 100 / 250 / 500 reference images

Usage:  python measure_cost.py [--n-test 200] [--device mps]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import config as cfg
from datasets import MVTecDataset
from run_experiments import build_method

METHODS = ["patchcore", "dino_patchcore_base", "spade", "padim", "clip_zs"]
REF_SIZES = [100, 250, 500]


def _bytes(obj) -> int:
    """Bytes of the fitted representation, excluding the frozen backbone."""
    total = 0
    for attr in ("memory_bank", "gallery", "mean", "inv_cov",
                 "_pos_feats", "_neg_feats"):
        v = getattr(obj, attr, None)
        if isinstance(v, np.ndarray):
            total += v.nbytes
        elif isinstance(v, torch.Tensor):
            total += v.element_size() * v.nelement()
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    ds = MVTecDataset(cfg.DATASET_PATHS["mvtec"], categories=cfg.MVTEC_CATEGORIES)
    train = ds.normal_train().image_paths
    test = ds.test().image_paths[: a.n_test]
    rows = []

    for name in METHODS:
        m = build_method(name, a.device)
        params = sum(p.numel() for p in getattr(m, "backbone", getattr(m, "model", torch.nn.Module())).parameters()) \
            if hasattr(m, "backbone") or hasattr(m, "model") else 0

        scaling = []
        for n_ref in REF_SIZES:
            mm = build_method(name, a.device)
            t0 = time.perf_counter()
            try:
                mm.fit(train[:n_ref] if name != "clip_zs" else [], seed=0)
            except ValueError as e:
                # Not a failure to paper over: PaDiM needs more reference images
                # than covariance dimensions, so it simply cannot run in the
                # smallest-reference regime. That is a scalability property
                # worth reporting, and one the memory-bank methods do not share.
                print(f"  {name:22s} n_ref={n_ref}: NOT APPLICABLE -- {e}")
                scaling.append({"n_ref": n_ref, "not_applicable": str(e)})
                continue
            t_fit = time.perf_counter() - t0
            mm.score(test[:8])                      # warm-up: first pass pays lazy init
            t0 = time.perf_counter()
            mm.score(test)
            t_score = time.perf_counter() - t0
            scaling.append({"n_ref": n_ref, "fit_s": round(t_fit, 2),
                            "stored_bytes": _bytes(mm),
                            "latency_ms": round(1000 * t_score / len(test), 2)})
            if name == "clip_zs":                   # reference-free: size is constant
                break

        r = {"method": name, "backbone_params_M": round(params / 1e6, 1), "scaling": scaling}
        rows.append(r)
        usable = [s for s in scaling if "not_applicable" not in s]
        if not usable:
            print(f"  {name:22s} no usable reference size"); continue
        big = usable[-1]
        print(f"  {name:22s} backbone {r['backbone_params_M']:6.1f}M params | "
              f"stored {big['stored_bytes']/1e6:8.2f} MB | fit {big['fit_s']:7.1f} s | "
              f"{big['latency_ms']:6.2f} ms/img")

    out = cfg.RESULTS_DIR / "cost.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"device": a.device, "n_test": len(test), "methods": rows}, open(out, "w"), indent=2)
    print(f"\n  written to {out}")


if __name__ == "__main__":
    main()
