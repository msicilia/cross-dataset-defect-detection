"""Binary defect masks for VISION, rasterised from its COCO polygons.

VISION annotates the defective images in train/ and val/ with polygon
segmentations (_annotations.coco.json in each directory). Each image's polygons
are rasterised at its native resolution, reduced to EVAL_SIZE x EVAL_SIZE by
area averaging, and every pixel whose 8-bit average is nonzero is marked as
defect, so defects narrower than one evaluation pixel are kept. EVAL_SIZE is the square
frame in which DINO-PatchCore loads images; its anomaly maps cover the centre
crop DINOPatchCore.map_box() of that frame.

Some COCO fields in this release are JSON-encoded strings; they are parsed
where present.

Masks are cached as PNG files under <vision_root>/.cache/masks_<EVAL_SIZE>/ and
built only when missing.

Usage:  python vision_masks.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import config as cfg

EVAL_SIZE = cfg.PATCHCORE["image_size"]


def _parse(value):
    return json.loads(value) if isinstance(value, str) else value


def _rasterise(polygons, width: int, height: int) -> np.ndarray:
    """Full-resolution binary mask from flat COCO polygons [x0, y0, x1, y1, ...]."""
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in polygons:
        pts = _parse(poly)
        if len(pts) >= 6:
            draw.polygon([(pts[i], pts[i + 1]) for i in range(0, len(pts) - 1, 2)],
                         fill=255)
    return np.array(canvas)


def build_masks(root: Path, categories: list[str] | None = None) -> dict[Path, Path]:
    """{image path: mask path} for every annotated defective image.

    Images whose annotations rasterise to an empty mask are omitted.
    """
    root = Path(root)
    cache = root / ".cache" / f"masks_{EVAL_SIZE}"
    categories = categories or sorted(p.name for p in root.iterdir()
                                      if p.is_dir() and not p.name.startswith("."))
    out: dict[Path, Path] = {}
    for cat in categories:
        for split in ("train", "val"):
            ann_file = root / cat / split / "_annotations.coco.json"
            if not ann_file.exists():
                continue
            coco = json.loads(ann_file.read_text())
            sizes = {int(i["id"]): (int(i["width"]), int(i["height"]), i["file_name"])
                     for i in coco["images"]}
            by_image: dict[int, list] = {}
            for ann in coco["annotations"]:
                seg = _parse(ann.get("segmentation") or "[]")
                if seg:
                    by_image.setdefault(int(_parse(ann["image_id"])), []).extend(seg)
            for img_id, polys in by_image.items():
                if img_id not in sizes:
                    continue
                w, h, fname = sizes[img_id]
                img_path = root / cat / split / fname
                if not img_path.exists():
                    continue
                mask_path = cache / cat / split / f"{Path(fname).stem}.png"
                if not mask_path.exists():
                    full = _rasterise(polys, w, h)
                    if not full.any():
                        continue
                    small = np.array(Image.fromarray(full).resize(
                        (EVAL_SIZE, EVAL_SIZE), Image.BOX))
                    if not (small > 0).any():
                        continue
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(((small > 0) * 255).astype(np.uint8)).save(mask_path)
                out[img_path] = mask_path
    return out


if __name__ == "__main__":
    masks = build_masks(cfg.DATASET_PATHS["vision"], cfg.VISION_CATEGORIES)
    areas = [np.mean(np.array(Image.open(p)) > 0) for p in masks.values()]
    print(f"{len(masks)} masks at {EVAL_SIZE}x{EVAL_SIZE}; defect area "
          f"mean {np.mean(areas) * 100:.2f}%, median {np.median(areas) * 100:.2f}% "
          f"of the image")
