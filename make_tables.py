"""Numeric tables of the paper and the quantities quoted in its text.

Every value is read from the outputs of the analysis scripts, which are the
single source of truth. The few values the paper derives from them by simple
averaging are computed here and marked where they are computed.

  file               label                   input
  datasets           tab:datasets            dataset directories (cfg.DATASET_PATHS)
  transfer_auroc     tab:transfer_auroc      significance.json
  gap                tab:gap                 significance.json, paradigm.json
  divergence         tab:divergence          divergence.json
  pooling            tab:pooling             per_category.json
  percat             tab:percat              per_category.json
  localisation       tab:localisation        localisation.json
  ablation_backbone  tab:ablation_backbone   significance.json
  mixed_source       tab:mixed_source        mixed_fewshot.json, significance.json
  cost               tab:cost                cost.json

tab:mapping holds no results and is not generated. Where two inputs carry the
same quantity (gaps in significance.json and paradigm.json, single-source
DINO-PatchCore AUROC in mixed_fewshot.json and significance.json) they must
agree, otherwise the inputs are stale and an error is raised.

Each <file>.tex holds the complete tabular environment, to be \\input inside
the paper's table float, which keeps caption, label and font size.

numbers.json maps self-explanatory keys to {"value", "description"} for the
quantities quoted in the text (unrounded); entries of the tables themselves are
not repeated there.

--compare PAPER extracts the numbers of each table from the paper source and
from the generated file and prints where they differ, aligned by position when
both contain the same count of numbers and by sequence alignment otherwise. It
is a report and exits with status 0.

Usage:
    python make_tables.py [--results-dir results] [--out results/tables]
                          [--compare ../paper/main.tex]
"""
from __future__ import annotations

import argparse
import difflib
import functools
import json
import os
import re
import statistics
import tempfile
from itertools import combinations
from pathlib import Path

import config as cfg
from common import build_dataset

DATASETS = cfg.DATASETS
REFERENCE_BASED = ["patchcore", "dino_patchcore_base", "spade", "padim", "winclip_plus"]
REFERENCE_FREE = ["winclip", "clip_zs"]
DETECTORS = REFERENCE_BASED + REFERENCE_FREE
METHOD_LABEL = {"patchcore": "PatchCore", "dino_patchcore_base": "DINO-PatchCore",
                "spade": "SPADE", "padim": "PaDiM", "winclip_plus": "WinCLIP+",
                "winclip": "WinCLIP", "clip_zs": "CLIP-ZS"}
BACKBONE_LABEL = {"dino_patchcore_small": "DINOv2-ViT-S/14",
                  "dino_patchcore_base": "DINOv2-ViT-B/14",
                  "dino_patchcore_large": "DINOv2-ViT-L/14"}
DATASET_LABEL = {"sdnet": "SDNET2018", "mvtec": "MVTec~AD", "vision": "VISION"}
SHORT_LABEL = {"sdnet": "SDNET", "mvtec": "MVTec", "vision": "VISION"}
SPLIT_GROUPS_SURFACE = ["sdnet", "mvtec_tex"]
SPLIT_GROUPS_METALLIC = ["mvtec_obj", "vision"]

# Row orders of the paper's tables.
DIVERGENCE_PAIRS = [("sdnet", "mvtec"), ("mvtec", "vision"), ("sdnet", "vision")]
DIVERGENCE_MEASURES = ["mmd2", "frechet", "cosine"]
LOCALISATION_ROWS = {"mvtec": ["mvtec", "sdnet", "vision"],
                     "vision": ["vision", "mvtec", "sdnet"]}
POOLING_TARGETS = ["mvtec", "vision"]
MIXED_PAIRS = [("sdnet", "mvtec"), ("sdnet", "vision"), ("mvtec", "vision")]

COST_REFERENCE_IMAGES = cfg.PATCHCORE["max_train_images"]
AGREEMENT_TOL = 1e-9
UNSTABLE_FACTOR = 2.0
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten"]

SIG = "significance.json"
PARADIGM = "paradigm.json"
PERCAT = "per_category.json"
INVERSION = "inversion.json"
LOCALISATION = "localisation.json"
DIVERGENCE = "divergence.json"
SPLIT = "mvtec_split.json"
MIXED = "mixed_fewshot.json"
CONTAMINATION = "contamination.json"
CONFOUNDERS = "confounders.json"
PROPORTIONS = "proportions.json"
COST = "cost.json"
SPADE_LAYERS = "spade_layers.json"


class InputError(RuntimeError):
    pass


class Inputs:
    """JSON outputs of the analysis scripts, loaded once, with checked access."""

    def __init__(self, results_dir: Path):
        self.dir = results_dir
        self._cache: dict[str, object] = {}

    def get(self, name: str, *keys):
        if name not in self._cache:
            path = self.dir / name
            if not path.is_file():
                raise InputError(f"missing input {path}; run the analysis script that writes it")
            self._cache[name] = json.loads(path.read_text())
        node = self._cache[name]
        for i, k in enumerate(keys):
            try:
                node = node[k]
            except (KeyError, IndexError, TypeError):
                trail = " > ".join(map(str, keys[:i + 1]))
                raise InputError(f"{self.dir / name}: missing key {trail}") from None
        if node is None:
            raise InputError(f"{self.dir / name}: {' > '.join(map(str, keys))} is null")
        return node


def agree(a: float, b: float, what: str) -> None:
    if abs(a - b) > AGREEMENT_TOL:
        raise InputError(f"inputs disagree on {what}: {a!r} vs {b!r}; re-run the analyses")


def sources_for(method: str) -> list[str]:
    return ["none"] if method in REFERENCE_FREE else list(DATASETS)


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def pm(mean: float, sd: float, digits: int = 3, bold: bool = False) -> str:
    body = f"{mean:.{digits}f}\\pm{sd:.{digits}f}"
    return f"$\\mathbf{{{body}}}$" if bold else f"${body}$"


def row(*cells: str) -> str:
    return "  " + " & ".join(cells) + r" \\"


def note(ncols: int, text: str, width: str = r"0.93\columnwidth") -> str:
    return f"  \\multicolumn{{{ncols}}}{{@{{}}p{{{width}}}@{{}}}}{{\\footnotesize {text}}} \\\\"


def tabular(spec: str, lines: list[str]) -> str:
    return "\n".join([f"\\begin{{tabular}}{{{spec}}}", *lines, r"\end{tabular}"]) + "\n"


def minimum_positions(formatted: list[str]) -> set[int]:
    """Indices of the smallest value after rounding (ties all count)."""
    values = [float(v) for v in formatted]
    return {i for i, v in enumerate(values) if v == min(values)}


def word(n: int) -> str:
    return NUMBER_WORDS[n] if n < len(NUMBER_WORDS) else str(n)


# ── tables ────────────────────────────────────────────────────────────────────

@functools.cache
def dataset_counts() -> dict:
    """Image and defect-type counts of the dataset releases on disk."""
    roots = {name: Path(cfg.DATASET_PATHS[name]) for name in DATASETS}
    for name, root in roots.items():
        if not root.is_dir():
            raise InputError(f"missing dataset directory {root} ({name})")

    def images(d: Path) -> int:
        return sum(1 for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    sdnet = images(roots["sdnet"])

    mv = roots["mvtec"]
    mv_all = sorted(p.name for p in mv.iterdir() if (p / "test").is_dir())
    missing = [c for c in cfg.MVTEC_CATEGORIES if c not in mv_all]
    if missing:
        raise InputError(f"{mv}: categories {missing} not found")

    def defect_types(cat: str) -> int:
        return sum(1 for d in (mv / cat / "test").iterdir() if d.is_dir() and d.name != "good")

    mv_images = sum(images(mv / c / s) for c in cfg.MVTEC_CATEGORIES for s in ("train", "test"))

    vi = roots["vision"]
    vi_images, vi_types = {}, {}
    for cat in cfg.VISION_CATEGORIES:
        listed, names = set(), set()
        for split in ("train", "val", "inference"):
            coco_path = vi / cat / split / "_annotations.coco.json"
            if not coco_path.is_file():
                raise InputError(f"missing {coco_path}")
            coco = json.loads(coco_path.read_text())
            listed |= {split + "/" + im["file_name"] for im in coco["images"]
                       if (vi / cat / split / im["file_name"]).exists()}
            if split != "inference":
                id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
                names |= {id_to_name[a["category_id"]] for a in coco["annotations"]}
        vi_images[cat] = len(listed)
        vi_types[cat] = len(names)

    loaders = {}
    for name in DATASETS:
        ds = build_dataset(name)
        test = ds.test()
        loaders[name] = {"reference": len(ds.normal_train().image_paths),
                         "test": len(test.image_paths), "test_defective": int(sum(test.labels))}
    return {
        "sdnet_images": sdnet,
        "mvtec_images": mv_images,
        "mvtec_defect_types": sum(defect_types(c) for c in cfg.MVTEC_CATEGORIES),
        "mvtec_categories_total": len(mv_all),
        "mvtec_defect_types_total": sum(defect_types(c) for c in mv_all),
        "vision_images_per_category": vi_images,
        "vision_images": sum(vi_images.values()),
        "vision_defect_types": sum(vi_types.values()),
        "loader_splits": loaders,
    }


def table_datasets() -> str:
    c = dataset_counts()
    mv_cats = ", ".join(x.replace("_", r"\_") for x in cfg.MVTEC_CATEGORIES)
    vi_cats = ", ".join(f"{k} ({v:,})" for k, v in c["vision_images_per_category"].items())
    head = (r"\textbf{Dataset} & \textbf{Domain} & \textbf{Images} & \textbf{Defect types}"
            r" & \textbf{Annotation} & \textbf{Licence} \\")
    lines = [
        r"\toprule", head, r"\midrule",
        row(r"SDNET2018~\cite{dorafshan2018sdnet}", "Concrete (walls, decks, pavements)",
            f"{c['sdnet_images']:,}", "2 (crack / intact)", "Image-level", r"CC~BY~4.0"),
        row(r"MVTec~AD~\cite{bergmann2021mvtec}", r"Industrial textures \& objects",
            f"{c['mvtec_images']:,}\\textsuperscript{{\\dag}}",
            f"{c['mvtec_defect_types']}\\textsuperscript{{\\dag}}", "Pixel mask", r"CC~BY-NC-SA~4.0"),
        row(r"VISION~\cite{bai2023vision}", r"Industrial mfg.\ (metallic components)",
            f"{c['vision_images']:,}\\textsuperscript{{*}}",
            f"{c['vision_defect_types']}\\textsuperscript{{*}}", "Polygon mask", r"CC~BY-NC~4.0"),
        r"\midrule",
        # StructDamage and UAV-Crack are not evaluated here; their entries, like the
        # licences above, are taken from the dataset publications. UAV-Crack's record
        # states no licence.
        row(r"StructDamage~\cite{structdamage2025}", "Concrete, asphalt, masonry", "78,093",
            r"--\textsuperscript{\ddag}", "Image-level", r"CC~BY~4.0"),
        row(r"UAV-Crack~\cite{yang2025uavcrack}", "Mixed surfaces (UAV-captured)", "195",
            "2 (crack / background)", "Segmentation", "--"),
        r"\bottomrule",
        f"  \\multicolumn{{6}}{{l}}{{\\footnotesize\\textsuperscript{{\\dag}}"
        f"{word(len(cfg.MVTEC_CATEGORIES)).capitalize()} of {c['mvtec_categories_total']} categories"
        f" used: {mv_cats} (image and defect-type counts are for these categories;"
        f" the full dataset has {c['mvtec_defect_types_total']} defect types).}} \\\\",
        # Defect types of the full VISION release, from the dataset publication
        # (only the configured subsets are on disk).
        f"  \\multicolumn{{6}}{{l}}{{\\footnotesize\\textsuperscript{{*}}"
        f"{word(len(cfg.VISION_CATEGORIES)).capitalize()} metallic subsets used: {vi_cats};"
        f" the full dataset has 44 defect types.}} \\\\",
        r"  \multicolumn{6}{l}{\footnotesize\textsuperscript{\ddag}StructDamage is organised by"
        r" surface type (nine) rather than by defect type.} \\",
    ]
    return tabular("llrrll", lines)


def table_transfer_auroc(J: Inputs) -> str:
    head = (r"\textbf{Method} & \textbf{Source $\backslash$ Target} & "
            + " & ".join(f"\\textbf{{{SHORT_LABEL[t]}}}" for t in DATASETS) + r" \\")
    lines = [r"\toprule", head, r"\midrule"]
    for m in REFERENCE_BASED:
        lines.append(f"  \\multirow{{{len(DATASETS)}}}{{*}}{{{METHOD_LABEL[m]}}}")
        for s in DATASETS:
            cells = []
            for t in DATASETS:
                cells.append(pm(J.get(SIG, "cells", f"{m}/{s}__{t}", "auroc_mean"),
                                J.get(SIG, "cells", f"{m}/{s}__{t}", "seed_sd"), bold=s == t))
            lines.append(row("", DATASET_LABEL[s], *cells))
        lines.append(r"\midrule")
    for m in REFERENCE_FREE:
        values = [f"{J.get(SIG, 'cells', f'{m}/none__{t}', 'auroc_mean'):.3f}" for t in DATASETS]
        lines.append(row(f"\\multicolumn{{2}}{{l}}{{{METHOD_LABEL[m]} (no source dataset needed)}}",
                         *values))
    lines.append(r"\bottomrule")
    return tabular("llccc", lines)


def check_paradigm(J: Inputs) -> None:
    rows = {r["method"]: r for r in J.get(PARADIGM, "detectors")}
    for m in DETECTORS:
        if m not in rows:
            raise InputError(f"{J.dir / PARADIGM}: no detector {m}")
        agree(J.get(SIG, "methods", m, "in_dist_mean"), rows[m]["in_distribution"],
              f"in-distribution AUROC of {m} (significance.json vs paradigm.json)")
        agree(J.get(SIG, "methods", m, "gap_mean"), rows[m]["gap"],
              f"gap of {m} (significance.json vs paradigm.json)")


def table_gap(J: Inputs) -> str:
    check_paradigm(J)
    ind = [f"{J.get(SIG, 'methods', m, 'in_dist_mean'):.3f}" for m in DETECTORS]
    gap = [f"{J.get(SIG, 'methods', m, 'gap_mean'):.3f}" for m in DETECTORS]
    gap_vf = [f"{J.get(SIG, 'methods', m, 'gap_vision_free_mean'):.3f}" for m in DETECTORS]
    # Seed-to-seed spread of the gap, from the per-seed values the tests use.
    # The reference-free detectors have no source dataset, hence no spread.
    sd = {k: {m: statistics.stdev(J.get(SIG, k, m)) for m in REFERENCE_BASED}
          for k in ("gaps", "gaps_vision_free")}
    # Lower is better: the smallest gap in each column is bold.
    best, best_vf = minimum_positions(gap), minimum_positions(gap_vf)
    lines = [r"\toprule",
             r"\textbf{Method} & \textbf{In-dist AUROC} & $\boldsymbol{\Delta}$ & "
             r"\mbox{$\boldsymbol{\Delta}$ excl.\ VISION} \\",
             r"\midrule"]
    for i, m in enumerate(DETECTORS):
        if m == REFERENCE_FREE[0]:
            lines.append(r"\midrule")
        if m in REFERENCE_BASED:
            g = pm(float(gap[i]), sd["gaps"][m])
            gv = pm(float(gap_vf[i]), sd["gaps_vision_free"][m])
        else:
            g = f"\\textbf{{{gap[i]}}}" if i in best else gap[i]
            gv = f"\\textbf{{{gap_vf[i]}}}" if i in best_vf else gap_vf[i]
        lines.append(row(METHOD_LABEL[m], ind[i], g, gv))
    lines += [r"\bottomrule", note(4, (
        r"The lower block uses no reference images. CLIP-ZS and WinCLIP achieve $\Delta=0$ by"
        r" construction, as they use no source dataset; this reflects domain-agnosticism, not"
        r" superior cross-domain transfer. WinCLIP and WinCLIP+ share a backbone, windows and"
        r" prompts, and differ only in whether reference images inform the score."))]
    return tabular("lccc", lines)


def divergence_key(J: Inputs, name: str, a: str, b: str, section: str) -> str:
    pairs = J.get(name, section)
    for key in (f"{a}__{b}", f"{b}__{a}"):
        if key in pairs:
            return key
    raise InputError(f"{J.dir / name}: no pair {a}/{b} in {section}")


def table_divergence(J: Inputs) -> str:
    keys = [divergence_key(J, DIVERGENCE, a, b, "pairs") for a, b in DIVERGENCE_PAIRS]
    means = {k: [f"{J.get(DIVERGENCE, 'pairs', p, k, 'mean'):.3f}" for p in keys]
             for k in DIVERGENCE_MEASURES}
    # Bold marks the closest pair under each measure.
    closest = {k: minimum_positions(v) for k, v in means.items()}
    lines = [r"\toprule",
             r"\textbf{Pair} & \textbf{MMD$^2$} & \textbf{Fr\'echet} & \textbf{Cosine} \\",
             r"\midrule"]
    for i, (p, (a, b)) in enumerate(zip(keys, DIVERGENCE_PAIRS)):
        cells = []
        for k in DIVERGENCE_MEASURES:
            mean = f"\\mathbf{{{means[k][i]}}}" if i in closest[k] else means[k][i]
            if k == "cosine":
                cells.append(f"${mean}$")
            else:
                cells.append(f"${mean}\\pm{J.get(DIVERGENCE, 'pairs', p, k, 'sd'):.3f}$")
        lines.append(row(f"{DATASET_LABEL[a]}--{DATASET_LABEL[b]}", *cells))
    lines += [r"\bottomrule", note(4, "Bold marks the closest pair under each measure. "
                                     + divergence_agreement(J, keys))]
    return tabular("lccc", lines)


def divergence_agreement(J: Inputs, keys: list[str]) -> str:
    """Sentence on which measures order the pairs identically (unrounded means)."""
    order = {k: sorted(keys, key=lambda p: J.get(DIVERGENCE, "pairs", p, k, "mean"))
             for k in DIVERGENCE_MEASURES}
    mmd_frechet = order["mmd2"] == order["frechet"]
    if mmd_frechet and order["mmd2"] == order["cosine"]:
        return r"All three measures order the pairs identically."
    if mmd_frechet:
        return r"MMD and Fr\'echet agree with each other and disagree with the cosine statistic."
    if order["mmd2"] == order["cosine"]:
        return r"MMD and the cosine statistic agree with each other and disagree with Fr\'echet."
    if order["frechet"] == order["cosine"]:
        return r"Fr\'echet and the cosine statistic agree with each other and disagree with MMD."
    return r"The three measures order the pairs differently."


def table_pooling(J: Inputs) -> str:
    lines = [
        r"\toprule",
        " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{\\textbf{{{DATASET_LABEL[t]} target}}}}"
                           for t in POOLING_TARGETS) + r" \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        " & " + " & ".join(r"\multicolumn{2}{c}{In-distribution} & \multicolumn{2}{c}{Cross-dataset}"
                           for _ in POOLING_TARGETS) + r" \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"\textbf{Method} & " + " & ".join([r"\textbf{Pooled} & \textbf{Macro}"] * 4) + r" \\",
        r"\midrule",
    ]
    for m in DETECTORS:
        if m == REFERENCE_FREE[0]:
            lines.append(r"\midrule")
        cells = []
        for t in POOLING_TARGETS:
            for cond in ("in_dist", "cross"):
                for conv in ("pooled", "macro"):
                    if cond == "in_dist" and m in REFERENCE_FREE:
                        cells.append("--")
                    else:
                        cells.append(f"${J.get(PERCAT, 'table', t, m, cond, conv):.3f}$")
        lines.append(row(METHOD_LABEL[m], *cells))
    lines += [r"\bottomrule", note(9, (
        r"WinCLIP and CLIP-ZS use no reference images, so the in-distribution/cross-dataset"
        r" distinction does not apply to them; their single value per target is shown in the"
        r" cross-dataset columns."), width=r"0.97\textwidth")]
    return tabular("lcccccccc", lines)


def categories(J: Inputs, target: str) -> list[str]:
    cats = None
    for m in DETECTORS:
        for s in sources_for(m):
            found = sorted(J.get(PERCAT, "cells", f"{m}/{s}__{target}", "per_category"))
            if cats is None:
                cats = found
            elif found != cats:
                raise InputError(f"{J.dir / PERCAT}: {m}/{s}__{target} has categories {found}, "
                                 f"expected {cats}")
    return cats


def per_category_value(J: Inputs, m: str, cond: str, t: str, c: str) -> float:
    if cond == "in_dist":
        return J.get(PERCAT, "cells", f"{m}/{t}__{t}", "per_category", c)
    if m in REFERENCE_FREE:
        return J.get(PERCAT, "cells", f"{m}/none__{t}", "per_category", c)
    # Cross-dataset: mean over the two foreign sources of the seed-mean category
    # AUROC, as per_category.json does for its pooled and macro table.
    foreign = [s for s in DATASETS if s != t]
    return sum(J.get(PERCAT, "cells", f"{m}/{s}__{t}", "per_category", c)
               for s in foreign) / len(foreign)


def table_percat(J: Inputs) -> str:
    cats = {t: categories(J, t) for t in POOLING_TARGETS}
    n = {t: len(cats[t]) for t in POOLING_TARGETS}
    first, second = POOLING_TARGETS
    lines = [
        r"\toprule",
        " & & " + " & ".join(f"\\multicolumn{{{n[t]}}}{{c}}{{\\textbf{{{DATASET_LABEL[t]}}}}}"
                             for t in POOLING_TARGETS) + r" \\",
        f"\\cmidrule(lr){{3-{2 + n[first]}}}\\cmidrule(lr){{{3 + n[first]}-{2 + n[first] + n[second]}}}",
        r" & \textbf{Method} & " + " & ".join(c.replace("_", r"\_")
                                             for t in POOLING_TARGETS for c in cats[t]) + r" \\",
        r"\midrule",
    ]
    for cond, members, name in (("in_dist", REFERENCE_BASED, "In-dist."),
                                ("cross", DETECTORS, "Cross-dataset")):
        if cond == "cross":
            lines.append(r"\midrule")
        lines.append(f"  \\multirow{{{len(members)}}}{{*}}{{\\rotatebox{{90}}{{{name}}}}}")
        for m in members:
            values = [f"${per_category_value(J, m, cond, t, c):.3f}$"
                      for t in POOLING_TARGETS for c in cats[t]]
            lines.append(row("", METHOD_LABEL[m], *values))
    lines.append(r"\bottomrule")
    return tabular("ll" + "c" * (n[first] + n[second]), lines)


def table_localisation(J: Inputs) -> str:
    lines = [r"\toprule",
             r"\textbf{Target} & \textbf{Source} & \textbf{Pixel AUROC} & \textbf{Peak rate}"
             r" & \textbf{$\times$ chance} \\",
             r"\midrule"]
    for i, (t, sources) in enumerate(LOCALISATION_ROWS.items()):
        if i:
            lines.append(r"\midrule")
        lines.append(f"  \\multirow{{{len(sources)}}}{{*}}{{{DATASET_LABEL[t]}}}")
        for s in sources:
            c = f"{s}__{t}"
            pix = pm(J.get(LOCALISATION, "cells", c, "pixel_auroc_defective", "mean"),
                     J.get(LOCALISATION, "cells", c, "pixel_auroc_defective", "sd"))
            rate = (f"${100 * J.get(LOCALISATION, 'cells', c, 'argmax_in_mask', 'mean'):.1f}"
                    f"\\pm{100 * J.get(LOCALISATION, 'cells', c, 'argmax_in_mask', 'sd'):.1f}\\%$")
            ratio = f"${J.get(LOCALISATION, 'cells', c, 'argmax_in_mask_x_chance'):.1f}$"
            lines.append(row("", DATASET_LABEL[s], pix, rate, ratio))
    chance = [f"\\mbox{{${100 * J.get(LOCALISATION, 'chance_level', t):.2f}\\%$}} on {DATASET_LABEL[t]}"
              for t in LOCALISATION_ROWS]
    lines += [r"\bottomrule", note(5, f"Chance for the peak rate is {' and '.join(chance)}."
                                     + localisation_stability(J))]
    return tabular("llccc", lines)


def localisation_stability(J: Inputs) -> str:
    """Sentence on the cell whose pixel AUROC varies most across seeds.

    A cell is called unstable when its pixel-AUROC seed SD is at least
    UNSTABLE_FACTOR times that of every other cell, and its peak rate stable
    when the peak-rate seed SD is below the median over cells. Empty when no
    cell stands out.
    """
    cells = list(J.get(LOCALISATION, "cells"))
    pix = {c: J.get(LOCALISATION, "cells", c, "pixel_auroc_defective", "sd") for c in cells}
    rate = {c: J.get(LOCALISATION, "cells", c, "argmax_in_mask", "sd") for c in cells}
    first, second = sorted(cells, key=pix.get, reverse=True)[:2]
    if pix[first] < UNSTABLE_FACTOR * pix[second]:
        return ""
    s, t = first.split("__")
    peak = ("its peak rate is not" if rate[first] < statistics.median(rate.values())
            else "so is its peak rate")
    return (f" The pixel AUROC of the \\mbox{{{DATASET_LABEL[s]}$\\rightarrow${DATASET_LABEL[t]}}}"
            f" cell is unstable across seeds; {peak}.")


def table_ablation_backbone(J: Inputs) -> str:
    lines = [r"\toprule",
             r"\textbf{Backbone} & \textbf{In-dist} & \textbf{Cross-dataset} & $\boldsymbol{\Delta}$ \\",
             r"\midrule"]
    for m, label in BACKBONE_LABEL.items():
        b = ("backbone", "per_method", m)
        lines.append(row(label,
                         pm(J.get(SIG, *b, "in_dist_mean"), J.get(SIG, *b, "in_dist_sd")),
                         pm(J.get(SIG, *b, "cross_mean"), J.get(SIG, *b, "cross_sd")),
                         f"${J.get(SIG, *b, 'gap_mean'):.3f}$"))
    tests = J.get(SIG, "tests", "backbone_cross")
    max_t = max(abs(J.get(SIG, "tests", "backbone_cross", k, "t")) for k in tests)
    df = single(J, [J.get(SIG, "tests", "backbone_cross", k, "df") for k in tests], "backbone_cross df")
    # The three backbones are compared pairwise, so significance is read from the
    # Holm-adjusted p values rather than from the uncorrected critical value.
    min_p_holm = min(J.get(SIG, "tests", "backbone_cross", k, "p_holm") for k in tests)
    verdict = "No pairwise" if min_p_holm >= 0.05 else "At least one pairwise"
    lines += [r"\bottomrule", note(4, (
        f"{verdict} difference in the cross-dataset column is significant under paired"
        f" \\mbox{{$t$}}-tests across seeds with Holm correction for the three comparisons"
        f" \\mbox{{($|t|\\leq{max_t:.2f}$,}} \\mbox{{df $={df}$,}}"
        f" \\mbox{{smallest adjusted $p={min_p_holm:.2f}$).}}"), width=r"0.90\columnwidth")]
    return tabular("lccc", lines)


def single(J: Inputs, values: list, what: str):
    if len(set(values)) != 1:
        raise InputError(f"{J.dir}: {what} differs across entries: {sorted(set(values))}")
    return values[0]


def table_mixed_source(J: Inputs) -> str:
    base = ("mixed_source", "single_source")
    for s in DATASETS:
        for t in DATASETS:
            agree(J.get(MIXED, *base, s, t, "mean"),
                  J.get(SIG, "cells", f"dino_patchcore_base/{s}__{t}", "auroc_mean"),
                  f"DINO-PatchCore {s}->{t} (mixed_fewshot.json vs significance.json)")
    lines = [r"\toprule",
             r"\textbf{Source combination} & "
             + " & ".join(f"\\textbf{{{SHORT_LABEL[t]}}}" for t in DATASETS) + r" \\",
             r"\midrule"]
    for s in DATASETS:
        cells = []
        for t in DATASETS:
            v = f"{J.get(MIXED, *base, s, t, 'mean'):.3f}"
            cells.append(f"\\textbf{{{v}}}" if s == t else v)
        lines.append(row(f"{SHORT_LABEL[s]} (single)", *cells))
    lines.append(r"\midrule")
    for a, b in MIXED_PAIRS:
        key = f"{a}+{b}"
        cells = [pm(J.get(MIXED, "mixed_source", "mixed", key, t, "mean"),
                    J.get(MIXED, "mixed_source", "mixed", key, t, "sd")) for t in DATASETS]
        lines.append(row(f"{SHORT_LABEL[a]}+{SHORT_LABEL[b]}", *cells))
    lines.append(r"\bottomrule")
    return tabular("lccc", lines)


def cost_entry(J: Inputs, method: str) -> dict:
    """The cost.json measurement of a method at the benchmark bank size (single
    measurement for reference-free methods)."""
    entries = [e for e in J.get(COST, "methods") if e.get("method") == method]
    if len(entries) != 1:
        raise InputError(f"{J.dir / COST}: expected one entry for {method}, found {len(entries)}")
    scaling = entries[0].get("scaling") or []
    if method in REFERENCE_FREE:
        if len(scaling) != 1:
            raise InputError(f"{J.dir / COST}: {method} should have one measurement")
        rec = scaling[0]
    else:
        found = [r for r in scaling if r.get("n_ref") == COST_REFERENCE_IMAGES]
        if len(found) != 1:
            raise InputError(f"{J.dir / COST}: {method} has no measurement at "
                             f"n_ref={COST_REFERENCE_IMAGES}")
        rec = found[0]
    for k in ("stored_bytes", "fit_s", "latency_ms"):
        if rec.get(k) is None:
            raise InputError(f"{J.dir / COST}: {method} lacks {k} "
                             f"({rec.get('not_applicable', 'not recorded')})")
    return rec


def table_cost(J: Inputs) -> str:
    rec = {m: cost_entry(J, m) for m in DETECTORS}
    smallest_bank = min(REFERENCE_BASED, key=lambda m: rec[m]["stored_bytes"])
    lines = [r"\toprule",
             r"\textbf{Method} & \textbf{Stored} & \textbf{Fit} & \textbf{Latency} \\",
             r"\midrule"]
    for m in DETECTORS:
        mb = rec[m]["stored_bytes"] / 1e6
        stored = f"{mb:.1f}" if mb >= 1 else f"{mb:.2f}"
        # Bold marks the smallest storage among reference-based detectors.
        if m == smallest_bank:
            stored = f"\\textbf{{{stored}}}"
        fit = "--" if m in REFERENCE_FREE else f"{rec[m]['fit_s']:.0f}\\,s"
        ms = rec[m]["latency_ms"]
        latency = f"{ms:.0f}" if ms >= 100 else f"{ms:.1f}"
        lines.append(row(METHOD_LABEL[m], f"{stored}\\,MB", fit, f"{latency}\\,ms"))
    lines += [r"\bottomrule", note(4, (
        f"PaDiM requires at least \\mbox{{$\\approx${cfg.PADIM['n_dims']}}} reference images;"
        r" below that its per-position covariance is rank-deficient and the method cannot be"
        r" fitted."))]
    return tabular("lrrr", lines)


TABLES = {  # file name: (paper label, builder, inputs)
    "datasets": ("tab:datasets", lambda _: table_datasets(), "the dataset directories"),
    "transfer_auroc": ("tab:transfer_auroc", table_transfer_auroc, SIG),
    "gap": ("tab:gap", table_gap, f"{SIG} and {PARADIGM}"),
    "divergence": ("tab:divergence", table_divergence, DIVERGENCE),
    "pooling": ("tab:pooling", table_pooling, PERCAT),
    "percat": ("tab:percat", table_percat, PERCAT),
    "localisation": ("tab:localisation", table_localisation, LOCALISATION),
    "ablation_backbone": ("tab:ablation_backbone", table_ablation_backbone, SIG),
    "mixed_source": ("tab:mixed_source", table_mixed_source, f"{MIXED} and {SIG}"),
    "cost": ("tab:cost", table_cost, COST),
}


# ── in-text quantities ────────────────────────────────────────────────────────

class Numbers:
    def __init__(self):
        self.items: dict[str, dict] = {}

    def add(self, key: str, value, description: str) -> None:
        if key in self.items:
            raise KeyError(f"duplicate number key {key}")
        self.items[key] = {"value": value, "description": description}


def span(values) -> list[float]:
    values = list(values)
    return [min(values), max(values)]


def collect_numbers(J: Inputs) -> dict:
    N = Numbers()
    label = METHOD_LABEL

    # detectors, gaps
    paradigm = {r["method"]: r for r in J.get(PARADIGM, "detectors")}
    cross = {}
    for m in DETECTORS:
        N.add(f"in_dist_auroc__{m}", J.get(SIG, "methods", m, "in_dist_mean"),
              f"{label[m]}: mean in-distribution AUROC over the three datasets")
        cross[m] = (J.get(SIG, "methods", m, "cross_mean") if m in REFERENCE_BASED
                    else paradigm[m]["cross_dataset"])
        N.add(f"cross_dataset_auroc__{m}", cross[m],
              f"{label[m]}: mean cross-dataset AUROC (reference-free: mean over targets)")
        N.add(f"gap__{m}", J.get(SIG, "methods", m, "gap_mean"), f"{label[m]}: generalisation gap")
        N.add(f"gap_excl_vision__{m}", J.get(SIG, "methods", m, "gap_vision_free_mean"),
              f"{label[m]}: generalisation gap on SDNET2018 and MVTec AD only")
    N.add("cross_dataset_auroc_span__detectors", max(cross.values()) - min(cross.values()),
          "max minus min cross-dataset AUROC over the seven detectors")
    nonzero = sorted((J.get(SIG, "methods", m, "gap_mean"), m) for m in REFERENCE_BASED)
    N.add("gap__smallest_reference_based", {"detector": nonzero[0][1], "gap": nonzero[0][0]},
          "smallest gap among reference-based detectors")
    N.add("gap__second_smallest_reference_based", {"detector": nonzero[1][1], "gap": nonzero[1][0]},
          "second smallest gap among reference-based detectors")

    # paired tests
    tests = J.get(SIG, "tests")
    for fam in tests:
        pairs = J.get(SIG, "tests", fam)
        N.add(f"ttest__{fam}__df", single(J, [pairs[p]["df"] for p in pairs], f"{fam} df"),
              f"paired t-tests ({fam}): degrees of freedom")
        N.add(f"ttest__{fam}__crit_0975",
              single(J, [pairs[p]["crit_0.975"] for p in pairs], f"{fam} critical value"),
              f"paired t-tests ({fam}): two-sided 5% critical value")
        for p in pairs:
            a, b = p.split("-")
            for k in ("t", "p", "p_holm", "mean_diff"):
                N.add(f"ttest__{fam}__{a}__vs__{b}__{k}", J.get(SIG, "tests", fam, p, k),
                      f"paired t-test ({fam}), {a} minus {b}: {k}")

    def t_of(fam: str, a: str, b: str) -> float:
        pairs = J.get(SIG, "tests", fam)
        if f"{a}-{b}" in pairs:
            return pairs[f"{a}-{b}"]["t"]
        if f"{b}-{a}" in pairs:
            return -pairs[f"{b}-{a}"]["t"]
        raise InputError(f"{J.dir / SIG}: no test {a}-{b} in {fam}")

    trio = ["patchcore", "dino_patchcore_base", "padim"]
    for fam in ("paradigm_full", "paradigm_vision_free"):
        N.add(f"ttest__{fam}__max_abs_t__patchcore_dino_patchcore_base_padim",
              max(abs(t_of(fam, a, b)) for a, b in combinations(trio, 2)),
              f"largest |t| among PatchCore, DINO-PatchCore and PaDiM ({fam})")
        N.add(f"ttest__{fam}__min_abs_t__winclip_plus_vs_other_reference_based",
              min(abs(t_of(fam, "winclip_plus", m)) for m in REFERENCE_BASED if m != "winclip_plus"),
              f"smallest |t| of WinCLIP+ against the other reference-based detectors ({fam})")
    N.add("ttest__backbone_cross__max_abs_t",
          max(abs(v["t"]) for v in J.get(SIG, "tests", "backbone_cross").values()),
          "largest |t| between DINOv2 backbones, cross-dataset AUROC")
    N.add("ttest__backbone_in_dist__min_abs_t",
          min(abs(v["t"]) for v in J.get(SIG, "tests", "backbone_in_dist").values()),
          "smallest |t| between DINOv2 backbones, in-distribution AUROC")

    # backbones
    per = {m: J.get(SIG, "backbone", "per_method", m) for m in BACKBONE_LABEL}
    N.add("cross_dataset_auroc_span__backbones",
          max(v["cross_mean"] for v in per.values()) - min(v["cross_mean"] for v in per.values()),
          "max minus min cross-dataset AUROC over the three DINOv2 backbones")
    ind = [per[m]["in_dist_mean"] for m in BACKBONE_LABEL]
    N.add("backbone__in_dist_decreases_with_size", all(x > y for x, y in zip(ind, ind[1:])),
          "in-distribution AUROC strictly decreases from ViT-S to ViT-B to ViT-L")
    for key, what in (("lowest_cross_per_seed", "lowest cross-dataset"),
                      ("highest_in_dist_per_seed", "highest in-distribution")):
        winners = J.get(SIG, "backbone", key)
        for m in BACKBONE_LABEL:
            N.add(f"backbone__n_seeds_{key.replace('_per_seed', '')}__{m}", winners.count(m),
                  f"seeds in which {BACKBONE_LABEL[m]} has the {what} AUROC (of {len(winners)})")

    # interval widths
    w = ("interval_widths",)
    N.add("n_boot", J.get(SIG, *w, "n_boot"), "bootstrap resamples per cell")
    N.add("seed_sd_mean__reference_based_cells", J.get(SIG, *w, "seed_sd_mean"),
          "mean seed SD over the reference-based cells")
    N.add("n_cells__reference_based", len(J.get(SIG, *w, "seed_cells")),
          "number of reference-based cells (detectors x sources x targets)")
    N.add("bootstrap_ci_width_mean__all_cells", J.get(SIG, *w, "bootstrap_mean_width"),
          "mean 95% bootstrap interval width over all cells (reference-based and reference-free)")
    N.add("n_cells__bootstrap", len(J.get(SIG, *w, "bootstrap_cells")),
          "number of cells entering bootstrap_ci_width_mean__all_cells")
    N.add("bootstrap_ci_width_mean__reference_based_cells",
          J.get(SIG, *w, "bootstrap_mean_width_reference_based"),
          "mean 95% bootstrap interval width over the reference-based cells")
    N.add("seed_interval_width_mean__reference_based_cells", J.get(SIG, *w, "seed_interval_mean_width"),
          "mean 95% t-interval width about the seed mean implied by the seed SD")
    N.add("ratio_bootstrap_to_seed_interval_width",
          J.get(SIG, *w, "ratio_bootstrap_to_seed_reference_based"),
          "reference-based bootstrap width divided by seed-interval width")
    for t in DATASETS:
        N.add(f"bootstrap_ci_width_mean__target__{t}",
              J.get(SIG, *w, "bootstrap_mean_width_by_target", t, "mean_width"),
              f"mean 95% bootstrap interval width over cells with target {t}")
        N.add(f"n_test_images__{t}",
              single(J, J.get(SIG, *w, "bootstrap_mean_width_by_target", t, "n_test"), f"n_test {t}"),
              f"test images of {t}")
        N.add(f"n_test_defective__{t}", J.get(SIG, "cells", f"clip_zs/none__{t}", "n_test_defective"),
              f"defective test images of {t}")
    for s in DATASETS:
        if s != "vision":
            c = f"patchcore/{s}__vision"
            N.add(f"bootstrap_ci__patchcore__{s}__vision",
                  [J.get(SIG, "cells", c, "boot_ci_low"), J.get(SIG, "cells", c, "boot_ci_high")],
                  f"95% bootstrap interval of PatchCore {s}->vision (seed 0)")

    # inversion
    inv = ("summary",)
    cells = J.get(INVERSION, "cells")
    inverted = J.get(INVERSION, *inv, "inverted_cells")
    N.add("inversion__threshold", J.get(INVERSION, "config", "threshold"),
          "mean AUROC below which a cell counts as inverted")
    N.add("inversion__n_cells", J.get(INVERSION, *inv, "n_inverted"), "number of inverted cells")
    N.add("inversion__cells", inverted, "inverted cells")
    N.add("inversion__n_negative_cohens_d", J.get(INVERSION, *inv, "n_inverted_negative_d"),
          "inverted cells with negative Cohen's d")
    N.add("inversion__cohens_d_range", J.get(INVERSION, *inv, "cohens_d_range_inverted"),
          "[min, max] Cohen's d over inverted cells")
    N.add("inversion__norm_shift_range_with_source",
          J.get(INVERSION, *inv, "norm_shift_range_inverted_with_source"),
          "[min, max] normalised score shift over inverted cells that have a source")
    N.add("inversion__n_cells_with_source", J.get(INVERSION, *inv, "n_inverted_with_source"),
          "inverted cells of reference-based detectors")
    N.add("inversion__n_cells_involving_vision", J.get(INVERSION, *inv, "inverted_involving_vision"),
          "inverted cells with VISION as source or target")
    N.add("inversion__n_detectors", len({cells[k]["method"] for k in inverted}),
          "detectors with at least one inverted cell")
    N.add("inversion__highest_auroc_inverted", max(cells[k]["auroc_mean"] for k in inverted),
          "largest mean AUROC among inverted cells")
    N.add("inversion__lowest_auroc_not_inverted",
          J.get(INVERSION, *inv, "lowest_auroc_not_inverted", "auroc_mean"),
          "smallest mean AUROC among cells not inverted; every threshold above"
          " inversion__highest_auroc_inverted and up to this value selects the cells in"
          " inversion__cells")
    N.add("inversion__lowest_auroc_not_inverted_cell",
          J.get(INVERSION, *inv, "lowest_auroc_not_inverted", "cell"),
          "cell with the smallest mean AUROC among cells not inverted")
    N.add("norm_shift__dino_patchcore_base__sdnet__vision",
          J.get(INVERSION, "cells", "dino_patchcore_base/sdnet__vision", "norm_shift_mean"),
          "normalised score shift of DINO-PatchCore sdnet->vision")
    for k in ("auroc_mean", "cohens_d_mean"):
        N.add(f"winclip__vision__{k.replace('_mean', '')}",
              J.get(INVERSION, "cells", "winclip/none__vision", k), f"WinCLIP on VISION: {k}")

    # SPADE feature layers
    for arm, what in (("spade_layers", "SPADE's layer2-layer4"),
                      ("patchcore_layers", "PatchCore's layer2-layer3")):
        for stat in ("mean", "sd"):
            N.add(f"spade_layer_ablation__in_dist_auroc_{stat}__{arm}",
                  J.get(SPADE_LAYERS, "in_dist_auroc", arm, stat),
                  f"SPADE in-distribution on MVTec AD with {what}: {stat} over seeds")
    for stat in ("mean", "sd"):
        N.add(f"spade_layer_ablation__loss_from_patchcore_layers_{stat}",
              J.get(SPADE_LAYERS, "loss_from_restricting_to_patchcore_layers", stat),
              f"in-distribution AUROC SPADE loses on MVTec AD when restricted to"
              f" PatchCore's two layers: {stat} over seeds")

    # pooled versus macro
    for t in POOLING_TARGETS:
        for group in ("reference_based", "all"):
            g = ("macro_vs_pooled", t, group)
            N.add(f"macro_gt_pooled__{t}__{group}__count", J.get(PERCAT, *g, "n_macro_greater"),
                  f"table entries on {t} ({group}) with macro > pooled")
            N.add(f"macro_gt_pooled__{t}__{group}__entries", J.get(PERCAT, *g, "n_entries"),
                  f"table entries on {t} ({group})")
            exc = J.get(PERCAT, *g, "exceptions")
            N.add(f"macro_gt_pooled__{t}__{group}__exceptions",
                  [{"method": e["method"], "condition": e["condition"],
                    "macro_minus_pooled": e["macro_minus_pooled"]} for e in exc],
                  f"entries on {t} ({group}) with macro <= pooled")
        for cond in ("in_dist", "cross"):
            N.add(f"ordering_identical__{t}__{cond}",
                  J.get(PERCAT, "orderings", t, cond, "identical"),
                  f"detector ordering on {t} ({cond}) identical under pooled and macro AUROC")
    for m in REFERENCE_BASED:
        for conv in ("pooled", "macro"):
            N.add(f"{conv}_auroc__{m}__vision__mvtec",
                  J.get(PERCAT, "cells", f"{m}/vision__mvtec", f"{conv}_mean"),
                  f"{label[m]} vision->mvtec {conv} AUROC")

    # localisation
    chance = {t: J.get(LOCALISATION, "chance_level", t) for t in LOCALISATION_ROWS}
    for t in LOCALISATION_ROWS:
        N.add(f"localisation__chance__{t}", chance[t], f"peak-rate chance level (mean mask area) on {t}")
        for k in ("n_defective_scored", "n_defective_outside_crop"):
            N.add(f"localisation__{k}__{t}",
                  single(J, [J.get(LOCALISATION, "cells", f"{s}__{t}", k) for s in LOCALISATION_ROWS[t]],
                         f"{k} {t}"), f"{k} on target {t}")
    N.add("localisation__chance_ratio__mvtec_to_vision", chance["mvtec"] / chance["vision"],
          "chance level on MVTec AD divided by that on VISION")

    # divergence
    for p in J.get(DIVERGENCE, "pairs"):
        for k in DIVERGENCE_MEASURES + ["separability"]:
            N.add(f"divergence__{k}__{p}", J.get(DIVERGENCE, "pairs", p, k, "mean"),
                  f"{k} between {p.replace('__', ' and ')} (mean over draws)")
    pairs = J.get(DIVERGENCE, "pairs")
    N.add("divergence__n_per_dataset", J.get(DIVERGENCE, "config", "n_per_dataset"),
          "images per dataset per draw")
    N.add("divergence__n_draws", len(J.get(DIVERGENCE, "config", "draws")), "number of draws")
    N.add("divergence__separability_min", min(v["separability"]["mean"] for v in pairs.values()),
          "smallest cross-validated separability accuracy over pairs")
    for k in DIVERGENCE_MEASURES:
        means = sorted(v[k]["mean"] for v in pairs.values())
        N.add(f"divergence__closest_pair__{k}", min(pairs, key=lambda p: pairs[p][k]["mean"]),
              f"closest pair by {k}")
        N.add(f"divergence__farthest_pair__{k}", max(pairs, key=lambda p: pairs[p][k]["mean"]),
              f"most distant pair by {k}")
        N.add(f"divergence__max_sd_over_min_pair_difference__{k}",
              max(v[k]["sd"] for v in pairs.values()) / min(b - a for a, b in zip(means, means[1:])),
              f"largest draw SD divided by the smallest difference between pair means ({k})")

    # MVTec split
    groups = J.get(SPLIT, "groups")
    fp = J.get(SPLIT, "feature_pairs")
    for p in fp:
        N.add(f"mvtec_split__cosine__{p}", J.get(SPLIT, "feature_pairs", p, "cosine", "mean"),
              f"mean pairwise cosine distance between {p.replace('__', ' and ')}")
    N.add("mvtec_split__closest_pair__cosine", min(fp, key=lambda p: fp[p]["cosine"]["mean"]),
          "closest group pair by cosine distance")
    N.add("mvtec_split__farthest_pair__cosine", max(fp, key=lambda p: fp[p]["cosine"]["mean"]),
          "most distant group pair by cosine distance")
    N.add("mvtec_split__n_per_group", J.get(SPLIT, "config", "n_per_group"),
          "images per group per draw (feature distance)")
    N.add("mvtec_split__n_draws", len(J.get(SPLIT, "config", "draws")),
          "number of draws (feature distance)")
    mean, sd = J.get(SPLIT, "transfer_auroc"), J.get(SPLIT, "transfer_auroc_sd")
    tr = {(s, t): (mean[i][j], sd[i][j]) for i, s in enumerate(groups) for j, t in enumerate(groups)}
    for (s, t), (mu, sigma) in tr.items():
        N.add(f"mvtec_split__transfer__{s}__{t}", {"mean": mu, "sd": sigma},
              f"DINO-PatchCore AUROC {s}->{t}, mean and SD over seeds")
    for k in J.get(SPLIT, "transfer_averages"):
        N.add(f"mvtec_split__transfer_average__{k}", J.get(SPLIT, "transfer_averages", k, "mean"),
              f"mean off-diagonal transfer AUROC, {k.replace('_', ' ')}")
    N.add("mvtec_split__transfer_range__metallic_to_surface",
          span(tr[(s, t)][0] for s in SPLIT_GROUPS_METALLIC for t in SPLIT_GROUPS_SURFACE),
          "[min, max] AUROC from metallic sources to surface targets")
    N.add("mvtec_split__transfer_range__surface_to_vision",
          span(tr[(s, "vision")][0] for s in SPLIT_GROUPS_SURFACE),
          "[min, max] AUROC from surface sources to VISION")

    # confounders
    arms = J.get(CONFOUNDERS, "arms")
    base_gap = J.get(CONFOUNDERS, "arms", "baseline", "gap", "mean")
    N.add("confounders__n_seeds", len(J.get(CONFOUNDERS, "seeds")), "seeds per confounder arm")
    for arm in arms:
        for k in ("gap", "in_distribution", "cross_dataset"):
            N.add(f"confounders__{k}__{arm}", J.get(CONFOUNDERS, "arms", arm, k, "mean"),
                  f"DINO-PatchCore {k.replace('_', ' ')} under the {arm} arm")
        if arm != "baseline":
            N.add(f"confounders__gap_change_vs_baseline__{arm}",
                  J.get(CONFOUNDERS, "arms", arm, "gap", "mean") - base_gap,
                  f"gap under {arm} minus baseline gap")
    N.add("confounders__gap_range__normalised_arms",
          span(J.get(CONFOUNDERS, "arms", a, "gap", "mean") for a in arms if a != "baseline"),
          "[min, max] gap over the normalisation arms")

    # mixed sources
    mixed = J.get(MIXED, "mixed_source", "mixed")
    deltas = []
    for key in mixed:
        values = [J.get(MIXED, "mixed_source", "mixed", key, t, "minus_best_single") for t in DATASETS]
        deltas += values
        N.add(f"mixed__minus_best_single__{key}__range", span(values),
              f"[min, max] over targets of the {key} bank AUROC minus the better single source")
    N.add("mixed__minus_best_single__max", max(deltas),
          "largest mixed-bank AUROC minus better single source over all banks and targets")

    # contamination
    summary = J.get(CONTAMINATION, "summary")
    for k in summary:
        if k != "figure_cases":
            N.add(f"contamination__{k}", summary[k], f"[min, max] {k.replace('_', ' ')}")
    for case in J.get(CONTAMINATION, "summary", "figure_cases"):
        conds = ("cells", case, "conditions")
        name = case.replace("->", "__")
        for c in ("baseline", "plus_same", "plus_vision"):
            N.add(f"contamination__{name}__{c}", J.get(CONTAMINATION, *conds, c, "mean"),
                  f"AUROC {case}, {c} bank (mean over seeds)")
        N.add(f"contamination__{name}__plus_vision_minus_plus_same",
              {"mean": J.get(CONTAMINATION, *conds, "plus_vision", "delta_vs_plus_same", "mean"),
               "sd": J.get(CONTAMINATION, *conds, "plus_vision", "delta_vs_plus_same", "sd")},
              f"per-seed AUROC difference plus_vision minus plus_same, {case}")

    # proportions
    bases = J.get(PROPORTIONS, "bases")
    for b in bases:
        for t in J.get(PROPORTIONS, "bases", b):
            for pt in J.get(PROPORTIONS, "bases", b, t):
                N.add(f"proportions__{b}__{t}__vision{pt['vision_pct']:03d}", pt["mean"],
                      f"AUROC {b} base -> {t} with {pt['vision_pct']}% VISION in the bank")

    # reference-set size
    fs = ("fewshot",)
    n_values = J.get(MIXED, *fs, "n_values")
    N.add("fewshot__n_values", n_values, "reference-set sizes")
    for s in J.get(MIXED, *fs, "curves"):
        points = J.get(MIXED, *fs, "curves", s)
        for metric in ("cross_dataset", "in_distribution"):
            means = [p[metric]["mean"] for p in points]
            N.add(f"fewshot__{s}__{metric}__n{points[0]['n']}", means[0],
                  f"DINO-PatchCore {metric.replace('_', ' ')} AUROC from {s}, smallest n")
            N.add(f"fewshot__{s}__{metric}__n{points[-1]['n']}", means[-1],
                  f"DINO-PatchCore {metric.replace('_', ' ')} AUROC from {s}, largest n")
            N.add(f"fewshot__{s}__{metric}__range", span(means),
                  f"[min, max] over n of {metric.replace('_', ' ')} AUROC from {s}")

    # cost
    rec = {m: cost_entry(J, m) for m in DETECTORS}
    N.add("cost__device", J.get(COST, "device"), "device used for the cost measurements")
    N.add("cost__reference_images", COST_REFERENCE_IMAGES, "reference images for the cost table")
    for m in DETECTORS:
        N.add(f"cost__stored_mb__{m}", rec[m]["stored_bytes"] / 1e6, f"{label[m]} storage (MB)")
        N.add(f"cost__fit_s__{m}", rec[m]["fit_s"], f"{label[m]} fit time (s)")
        N.add(f"cost__latency_ms__{m}", rec[m]["latency_ms"], f"{label[m]} latency (ms per image)")
    stored = {m: rec[m]["stored_bytes"] for m in DETECTORS}
    N.add("cost__stored_ratio__patchcore_to_dino_patchcore_base",
          stored["patchcore"] / stored["dino_patchcore_base"], "PatchCore / DINO-PatchCore storage")
    N.add("cost__stored_ratio__winclip_plus_to_winclip", stored["winclip_plus"] / stored["winclip"],
          "WinCLIP+ / WinCLIP storage")
    largest = sorted(stored.values(), reverse=True)
    N.add("cost__stored_ratio__largest_to_second_largest", largest[0] / largest[1],
          f"largest / second largest storage (largest: {max(stored, key=stored.get)})")
    padim = [e for e in J.get(COST, "methods") if e.get("method") == "padim"][0]
    N.add("cost__padim_not_applicable_n_ref",
          [r["n_ref"] for r in padim["scaling"] if "not_applicable" in r],
          "reference-set sizes at which PaDiM could not be fitted")

    # datasets
    c = dataset_counts()
    for k in ("sdnet_images", "mvtec_images", "vision_images", "mvtec_defect_types",
              "vision_defect_types", "mvtec_categories_total", "mvtec_defect_types_total"):
        N.add(f"dataset__{k}", c[k], k.replace("_", " ") + " (used categories unless 'total')")
    for name, v in c["loader_splits"].items():
        for k, n in v.items():
            N.add(f"dataset__{name}__{k}_images", n, f"{name}: {k} images as returned by the loader")
    return N.items


# ── comparison with the paper ─────────────────────────────────────────────────

NUMBER = re.compile(r"(?<![\w.,/\-])-?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+(?:\.\d+)?)(?![\w])")


def skip_group(s: str, i: int) -> int:
    """Index after the brace group starting at s[i] (whitespace skipped)."""
    while i < len(s) and s[i].isspace():
        i += 1
    if i >= len(s) or s[i] != "{":
        raise ValueError(f"expected a brace group at: {s[i:i + 40]!r}")
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return j + 1
    raise ValueError("unbalanced braces")


def drop_command(s: str, name: str, groups: int) -> str:
    """Remove \\name with its optional (...) argument and `groups` brace groups."""
    out, i = [], 0
    pattern = re.compile(r"\\" + name + r"(?![A-Za-z])")
    while (m := pattern.search(s, i)):
        out.append(s[i:m.start()])
        j = m.end()
        if s[j:j + 1] == "(":
            j = s.index(")", j) + 1
        for _ in range(groups):
            j = skip_group(s, j)
        i = j
    out.append(s[i:])
    return "".join(out)


def clean(s: str) -> str:
    s = re.sub(r"\\[A-Za-z]+\*?", " ", s)
    s = re.sub(r"\\.", " ", s)
    return re.sub(r"\s+", " ", re.sub(r"[{}$~^_&]", " ", s)).strip()


TABULAR_BEGIN = re.compile(r"\\begin\{tabular(\*?)\}")


def tabular_numbers(tab: str) -> list[tuple[str, str]]:
    """(number, row label) for every number in a tabular or tabular* environment."""
    tab = re.sub(r"(?<!\\)%.*", "", tab)
    m = TABULAR_BEGIN.search(tab)
    if m is None:
        raise ValueError("no tabular environment")
    start = m.end()
    # tabular* takes a width group before the column spec.
    for _ in range(2 if m.group(1) else 1):
        start = skip_group(tab, start)
    body = tab[start:tab.index(f"\\end{{tabular{m.group(1)}}}")]
    for name, groups in (("multicolumn", 2), ("multirow", 2), ("cmidrule", 1), ("rotatebox", 1),
                         ("setlength", 2), ("cite", 1), ("ref", 1), ("label", 1)):
        body = drop_command(body, name, groups)
    found, group = [], ""
    for line in re.split(r"\\\\", body):
        cells = [clean(c) for c in re.split(r"(?<!\\)&", line)]
        if cells and cells[0]:
            group = cells[0]
        label = group
        if len(cells) > 1 and re.search(r"[A-Za-z]", cells[1]):
            label = f"{group} / {cells[1]}" if group else cells[1]
        for m in NUMBER.finditer(clean(line)):
            found.append((m.group().replace(",", ""), label[:48]))
    return found


def paper_tabular(tex: str, label: str) -> str:
    at = tex.find(f"\\label{{{label}}}")
    if at < 0:
        raise ValueError(f"\\label{{{label}}} not found")
    m = TABULAR_BEGIN.search(tex, at)
    end_float = tex.find(r"\end{table", at)
    if m is None or m.start() > end_float:
        raise ValueError(f"no tabular after \\label{{{label}}}")
    end = f"\\end{{tabular{m.group(1)}}}"
    return tex[m.start():tex.index(end, m.start()) + len(end)]


def compare(paper: Path, generated: dict[str, str]) -> None:
    tex = paper.read_text()
    print(f"\nNumbers in the tables of {paper} versus the generated tables")
    for name, (label, _, _) in TABLES.items():
        try:
            paper_nums = tabular_numbers(paper_tabular(tex, label))
        except ValueError as e:
            print(f"\n== {name} ({label}): cannot read the paper table: {e}")
            continue
        gen_nums = tabular_numbers(generated[name])
        ov, nv = [v for v, _ in paper_nums], [v for v, _ in gen_nums]
        lines = []
        if len(ov) == len(nv):
            lines = [f"  [{ol}] {o} -> {n}" for (o, ol), (n, _) in zip(paper_nums, gen_nums) if o != n]
            summary = f"{len(ov)} numbers, {len(lines)} differ"
        else:
            matcher = difflib.SequenceMatcher(None, ov, nv, autojunk=False)
            for op, i1, i2, j1, j2 in matcher.get_opcodes():
                if op == "equal":
                    continue
                pairs = max(i2 - i1, j2 - j1)
                for k in range(pairs):
                    o = paper_nums[i1 + k] if i1 + k < i2 else None
                    n = gen_nums[j1 + k] if j1 + k < j2 else None
                    where = (o or n)[1]
                    lines.append(f"  [{where}] {o[0] if o else '(none)'} -> {n[0] if n else '(none)'}")
            summary = f"{len(ov)} numbers in the paper, {len(nv)} generated, {len(lines)} differences"
        print(f"\n== {name} ({label}): {summary}")
        print("\n".join(lines) if lines else "  identical")


# ── output ────────────────────────────────────────────────────────────────────

def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR)
    ap.add_argument("--out", type=Path, default=None, help="default: <results-dir>/tables")
    ap.add_argument("--compare", type=Path, default=None, metavar="PAPER",
                    help="LaTeX source whose tables are compared with the generated ones")
    args = ap.parse_args()
    out = args.out or args.results_dir / "tables"

    J = Inputs(args.results_dir)
    generated = {name: build(J) for name, (_, build, _) in TABLES.items()}
    numbers = collect_numbers(J)

    for name, (label, _, source) in TABLES.items():
        header = f"% {label}: generated by make_tables.py from {source}.\n"
        write_atomic(out / f"{name}.tex", header + generated[name])
    write_atomic(out / "numbers.json", json.dumps(numbers, indent=2) + "\n")
    print(f"wrote {len(generated)} tables and {len(numbers)} numbers to {out}")

    if args.compare:
        compare(args.compare, generated)


if __name__ == "__main__":
    main()
