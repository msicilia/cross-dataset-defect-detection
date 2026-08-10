from __future__ import annotations
"""Build a normal-vs-defective sample montage across the three datasets.

Top row: defect-free (normal) examples; bottom row: defective examples.
Used in the paper to give readers an immediate visual sense of how different
the three defect domains are.

Output:
    results/figures/dataset_samples.pdf
    results/figures/dataset_samples.png   (for the plain-language docs)

Usage:
    python make_sample_montage.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import config as cfg

DATA    = cfg.DATA_ROOT
FIGURES = cfg.RESULTS_DIR / "figures"


def _first(directory: Path, pattern: str) -> Path | None:
    cands = [p for p in sorted(Path(directory).glob(pattern)) if not p.name.startswith("._")]
    return cands[0] if cands else None


def _defect_example(cat_test_dir: Path) -> Path | None:
    for sub in sorted(cat_test_dir.glob("*")):
        if sub.is_dir() and sub.name != "good":
            p = _first(sub, "*.png") or _first(sub, "*.jpg")
            if p:
                return p
    return None


def load(path: Path, size: int = 200) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((size, size)))


def main() -> None:
    mt = DATA / "mvtec_anomaly_detection"
    vs = DATA / "vision_dataset"
    columns = [
        ("SDNET2018\n(concrete)",
         _first(DATA / "sdnet2018" / "W" / "UW", "*.jpg"),
         _first(DATA / "sdnet2018" / "W" / "CW", "*.jpg")),
        ("MVTec AD\n(texture)",
         _first(mt / "tile" / "train" / "good", "*.png"),
         _defect_example(mt / "tile" / "test")),
        ("MVTec AD\n(object)",
         _first(mt / "metal_nut" / "train" / "good", "*.png"),
         _defect_example(mt / "metal_nut" / "test")),
        ("VISION\n(metallic)",
         _first(vs / "Casting" / "inference", "*.jpg"),
         _first(vs / "Casting" / "train", "*.jpg")),
    ]

    ncol = len(columns)
    fig, axes = plt.subplots(2, ncol, figsize=(2.0 * ncol, 4.4))
    for j, (title, normal, defect) in enumerate(columns):
        axes[0, j].imshow(load(normal))
        axes[0, j].set_title(title, fontsize=11)
        if defect is not None:
            axes[1, j].imshow(load(defect))
        else:
            axes[1, j].text(0.5, 0.5, "n/a", ha="center", va="center")
        for i in range(2):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
    axes[0, 0].set_ylabel("NORMAL", fontsize=11, fontweight="bold")
    axes[1, 0].set_ylabel("DEFECTIVE", fontsize=11, fontweight="bold", color="#b22222")

    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "dataset_samples.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "dataset_samples.png", dpi=140, bbox_inches="tight")
    print(f"→ wrote {FIGURES/'dataset_samples.pdf'} and .png")


if __name__ == "__main__":
    main()
