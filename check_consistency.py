"""Integrity checks for the result sets used in the paper.

For every result set under results/raw/, the number of complete cells (cells
with a result.json) is compared with the protocol, and every cell is checked:

  - result.json parses and records the cell's "config";
  - scores.npz holds scores, labels and paths of equal length, with finite
    scores;
  - every path belongs to the test split of a dataset as returned by its loader,
    with the same label; for sets scored on a whole target test split, paths
    and labels equal that split in order;
  - the stored image_auroc equals the AUROC recomputed from scores.npz.

Expected cell counts are derived from the protocol constants of the runner
scripts and config.py. Directories not listed in RESULT_SETS are ignored. Exits
with status 1 if any check fails.

Usage:  python check_consistency.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score

import config as cfg
import run_contamination
import run_proportions
from common import build_dataset
from run_confounders import ARMS
from run_experiments import REFERENCE_FREE as BENCHMARK_REFERENCE_FREE
from run_experiments import METHODS
from run_fewshot import N_SHOTS
from run_localisation import TARGETS as LOCALISATION_TARGETS
from ablate_spade_layers import ARMS as SPADE_LAYER_ARMS
from run_mixed_source import PAIRS
from run_mvtec_split import GROUPS

RAW = cfg.RESULTS_DIR / "raw"
EXAMPLES = 3            # problems printed per set and kind

N_DS = len(cfg.DATASETS)
N_SEEDS = len(cfg.SEEDS)
N_SUBSET_SEEDS = len(cfg.SUBSET_SEEDS)

# Banks per seed of the composition experiments, from the runners' own banks()
# applied to placeholder reference lists (only their lengths matter). In
# run_proportions a bank shared by several names is stored under each of them.
_PLACEHOLDER = {d: list(range(10_000)) for d in cfg.DATASETS}
CONTAMINATION_BANKS = len(run_contamination.banks(_PLACEHOLDER, 0))
PROPORTION_BANKS = sum(len(names) for names, _ in
                       run_proportions.banks(_PLACEHOLDER, 0).values())


@dataclass
class ResultSet:
    name: str
    cells: int
    full_split: bool = False    # every cell scores its whole target test split


REFERENCE_BASED = [m for m in METHODS if m not in BENCHMARK_REFERENCE_FREE]
REFERENCE_FREE = [m for m in METHODS if m in BENCHMARK_REFERENCE_FREE]

RESULT_SETS = [
    *(ResultSet(m, N_DS * N_DS * N_SEEDS, full_split=True) for m in REFERENCE_BASED),
    *(ResultSet(m, N_DS, full_split=True) for m in REFERENCE_FREE),
    ResultSet("dino_patchcore_mixed", len(PAIRS) * N_DS * N_SUBSET_SEEDS, full_split=True),
    ResultSet("dino_patchcore_mvtecsplit", len(GROUPS) ** 2 * N_SEEDS),
    ResultSet("dino_patchcore_fewshot",
              len(N_SHOTS) * N_DS * N_DS * N_SUBSET_SEEDS, full_split=True),
    ResultSet("dino_patchcore_contamination", CONTAMINATION_BANKS * N_DS * N_SEEDS),
    ResultSet("dino_patchcore_proportions", PROPORTION_BANKS * N_DS * N_SEEDS),
    ResultSet("dino_patchcore_confounders",
              len(ARMS) * N_DS * N_DS * len(cfg.CONFOUNDER_SEEDS)),
    ResultSet("dino_patchcore_localisation",
              N_DS * len(LOCALISATION_TARGETS) * N_SEEDS, full_split=True),
    ResultSet("spade_layers", len(SPADE_LAYER_ARMS) * N_SEEDS, full_split=True),
]


def load_test_splits() -> tuple[dict, dict]:
    """({dataset: (paths, labels)}, {path: label}) from the dataset loaders."""
    splits, label_of = {}, {}
    for name in cfg.DATASETS:
        test = build_dataset(name).test()
        paths = [str(p) for p in test.image_paths]
        splits[name] = (paths, list(test.labels))
        label_of.update(zip(paths, test.labels))
    return splits, label_of


def target_of(cell_rel) -> str | None:
    for part in reversed(cell_rel.parts):
        if "__" in part:
            return part.split("__", 1)[1]
    return None


def check_cell(cell, rs: ResultSet, splits: dict, label_of: dict) -> list[tuple[str, str]]:
    problems = []
    try:
        result = json.loads((cell / "result.json").read_text())
    except (OSError, ValueError) as e:
        return [("unreadable result.json", f"{type(e).__name__}")]
    if "config" not in result:
        problems.append(("result.json without config", ""))

    npz = cell / "scores.npz"
    if not npz.exists():
        return problems + [("missing scores.npz", "")]
    try:
        with np.load(npz, allow_pickle=False) as d:
            scores, labels, paths = d["scores"], d["labels"], [str(p) for p in d["paths"]]
    except Exception as e:
        return problems + [("unreadable scores.npz", f"{type(e).__name__}: {e}")]

    if not len(scores) == len(labels) == len(paths) > 0:
        return problems + [("scores/labels/paths lengths differ or are zero",
                            f"{len(scores)}/{len(labels)}/{len(paths)}")]
    if not np.isfinite(scores).all():
        return problems + [("non-finite scores", "")]

    unknown = [p for p in paths if p not in label_of]
    wrong = [p for p, y in zip(paths, labels) if p in label_of and label_of[p] != y]
    if unknown:
        problems.append(("paths not in any loader test split", f"{len(unknown)}, e.g. {unknown[0]}"))
    if wrong:
        problems.append(("labels differ from the loader", f"{len(wrong)}, e.g. {wrong[0]}"))
    if rs.full_split:
        tgt = target_of(cell.relative_to(RAW / rs.name))
        if tgt not in splits:
            problems.append(("cannot determine target test split", ""))
        elif (paths, labels.tolist()) != splits[tgt]:
            problems.append(("paths/labels differ from the full test split", tgt))

    if "image_auroc" not in result:
        problems.append(("result.json without image_auroc", ""))
    elif len(np.unique(labels)) < 2:
        problems.append(("single-class labels", ""))
    elif abs(result["image_auroc"] - roc_auc_score(labels, scores)) > 1e-12:
        problems.append(("stored image_auroc differs from recomputed AUROC",
                         f"{result['image_auroc']} vs {roc_auc_score(labels, scores)}"))
    return problems


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    splits, label_of = load_test_splits()
    n_problems = 0
    for rs in RESULT_SETS:
        root = RAW / rs.name
        cells = sorted(p.parent for p in root.rglob("result.json")) if root.exists() else []
        orphans = sorted({p.parent for p in root.rglob("scores.npz")} - set(cells)) \
            if root.exists() else []
        by_kind: dict[str, list[str]] = defaultdict(list)
        if len(cells) != rs.cells:
            by_kind["cell count"].append(f"{len(cells)} complete cells, expected {rs.cells}")
        for cell in orphans:
            by_kind["scores.npz without result.json"].append(str(cell.relative_to(root)))
        for cell in cells:
            for kind, detail in check_cell(cell, rs, splits, label_of):
                rel = str(cell.relative_to(root))
                by_kind[kind].append(f"{rel} {detail}".rstrip())

        status = "ok" if not by_kind else "FAIL"
        print(f"{rs.name:30s} {len(cells):4d}/{rs.cells:<4d} {status}")
        for kind, items in by_kind.items():
            n_problems += len(items)
            print(f"    {kind}: {len(items)}")
            for item in items[:EXAMPLES]:
                print(f"      {item}")
            if len(items) > EXAMPLES:
                print(f"      ... {len(items) - EXAMPLES} more")

    if n_problems:
        sys.exit(f"{n_problems} problem(s) found")
    print("all checks passed")


if __name__ == "__main__":
    main()
