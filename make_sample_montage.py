"""Montage of defect-free and defective images from the three datasets.

One column per dataset view (SDNET2018; MVTec AD tile and metal_nut; VISION
Casting). The top row is drawn from the reference split (normal_train) and the
bottom row from the defective test images: SDNET2018 cracked images, MVTec AD
images with a ground-truth mask, and VISION images with at least one COCO
annotation. Within each pool, candidates are sorted by path and one is chosen
with numpy default_rng(SEED), so the montage is deterministic.

Output: results/figures/dataset_samples.pdf, .png

Usage:
    python make_sample_montage.py [--results-dir results]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# TrueType, not matplotlib's default Type 3: Type 3 renders poorly at the sizes
# these figures are printed at, and IEEE does not accept it.
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config as cfg
from common import build_dataset

SEED = 0
SIZE = 200


def pick(paths: list[Path], what: str) -> Path:
    if not paths:
        raise RuntimeError(f"no candidate images for {what}")
    ordered = sorted(paths)
    return ordered[int(np.random.default_rng(SEED).integers(len(ordered)))]


def annotated_vision_images(split_dir: Path) -> set[Path]:
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    annotated = {a["image_id"] for a in coco["annotations"]}
    return {split_dir / im["file_name"] for im in coco["images"] if im["id"] in annotated}


def relative_to(root: Path, path: Path) -> tuple[str, ...]:
    return Path(path).relative_to(root).parts


def columns() -> list[tuple[str, Path, Path]]:
    sd, mv, vs = (build_dataset(d) for d in ("sdnet", "mvtec", "vision"))
    mv_root, vs_root = (cfg.DATASET_PATHS[d] for d in ("mvtec", "vision"))
    out = []

    test = sd.test()
    out.append(("SDNET2018\n(concrete)",
                pick(sd.normal_train().image_paths, "SDNET2018 normal"),
                pick([p for p, y in zip(test.image_paths, test.labels) if y == 1],
                     "SDNET2018 defective")))

    test = mv.test()
    for cat, title in (("tile", "MVTec AD\n(texture)"), ("metal_nut", "MVTec AD\n(object)")):
        normal = [p for p in mv.normal_train().image_paths
                  if relative_to(mv_root, p)[0] == cat]
        defective = [p for p, y, m in zip(test.image_paths, test.labels, test.mask_paths)
                     if y == 1 and relative_to(mv_root, p)[0] == cat
                     and m is not None and Path(m).is_file()]
        out.append((title, pick(normal, f"MVTec {cat} normal"),
                    pick(defective, f"MVTec {cat} defective")))

    cat = "Casting"
    annotated = set().union(*(annotated_vision_images(vs_root / cat / s)
                              for s in ("train", "val") if (vs_root / cat / s).is_dir()))
    test = vs.test()
    normal = [p for p in vs.normal_train().image_paths if relative_to(vs_root, p)[0] == cat]
    defective = [p for p, y in zip(test.image_paths, test.labels)
                 if y == 1 and relative_to(vs_root, p)[0] == cat and Path(p) in annotated]
    out.append(("VISION\n(metallic)", pick(normal, "VISION Casting normal"),
                pick(defective, "VISION Casting defective")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=cfg.RESULTS_DIR)
    args = ap.parse_args()

    cols = columns()
    fig, axes = plt.subplots(2, len(cols), figsize=(2.0 * len(cols), 4.4))
    for j, (title, normal, defective) in enumerate(cols):
        print(f"{title.replace(chr(10), ' ')}: normal {normal}, defective {defective}")
        for i, path in enumerate((normal, defective)):
            img = Image.open(path).convert("RGB").resize((SIZE, SIZE))
            axes[i, j].imshow(np.asarray(img))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[0, j].set_title(title, fontsize=11)
    axes[0, 0].set_ylabel("NORMAL", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("DEFECTIVE", fontsize=11, fontweight="bold", color="#b22222")
    fig.tight_layout()

    figures = args.results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(figures / f"dataset_samples.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {figures / 'dataset_samples.pdf'}")


if __name__ == "__main__":
    main()
