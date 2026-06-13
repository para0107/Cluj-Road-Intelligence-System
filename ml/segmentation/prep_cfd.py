"""
ml/segmentation/prep_cfd.py

PURPOSE:
    Converts CFD (Crack Forest Dataset) .seg files into PNG binary masks
    for SAM fine-tuning.

USED FOR:
    Task  : Instance segmentation training
    Model : SAM (Segment Anything Model)
    Why   : CFD provides pixel-level crack masks — exactly what SAM needs
            to fine-tune its mask decoder on road crack shapes.
            NOT used for RT-DETR detection training.

INPUT:
    data/datasets/cfd/CrackForest-dataset-master/
    ├── image/   001.jpg ... 118.jpg   (480x320 RGB)
    └── seg/     001.seg ... 118.seg   (run-length encoded masks)

.SEG FORMAT (each data line):
    label  row  col_start  col_end
    label=1 means crack pixel, label=0 means background.

OUTPUT:
    data/datasets/cfd/masks/     binary PNG masks (255=crack, 0=background)
    data/datasets/cfd/sam_train.json
    data/datasets/cfd/sam_val.json

SAM JSON ENTRY FORMAT:
    {
        "image":  "absolute/path/to/001.jpg",
        "mask":   "absolute/path/to/masks/001.png",
        "width":  480,
        "height": 320,
        "boxes":  [[x, y, w, h]]   <- SAM prompt box covering crack region
    }

USAGE:
    python ml/segmentation/prep_cfd.py
    python ml/segmentation/prep_cfd.py --val_ratio 0.2
"""

import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger

ROOT        = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "datasets" / "cfd" / "CrackForest-dataset-master"
IMAGES_DIR  = DATASET_DIR / "image"
SEG_DIR     = DATASET_DIR / "seg"
MASKS_DIR   = ROOT / "data" / "datasets" / "cfd" / "masks"
OUTPUT_DIR  = ROOT / "data" / "datasets" / "cfd"


def parse_seg_file(seg_path, width=480, height=320):
    """
    Parse .seg run-length file into a binary mask (numpy uint8 array).
    255 = crack pixel, 0 = background.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    in_data = False

    with open(seg_path, "r") as f:
        for line in f:
            line = line.strip()
            if line == "data":
                in_data = True
                continue
            if not in_data:
                if line.startswith("width"):
                    width = int(line.split()[1])
                    mask  = np.zeros((height, width), dtype=np.uint8)
                elif line.startswith("height"):
                    height = int(line.split()[1])
                    mask   = np.zeros((height, width), dtype=np.uint8)
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            label, row, c0, c1 = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if label == 1 and row < height:
                mask[row, c0:min(c1, width - 1) + 1] = 255

    return mask


def mask_to_box(mask):
    """
    Get a single bounding box [x, y, w, h] covering all crack pixels.
    Used as SAM prompt box — SAM refines the mask within this box.
    Returns [] if no crack pixels found.
    """
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any():
        return []
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return [[int(c0), int(r0), int(c1 - c0 + 1), int(r1 - r0 + 1)]]


def main(val_ratio=0.2, seed=42):
    logger.info("── CFD -> SAM segmentation format ─────────────────────────")
    MASKS_DIR.mkdir(parents=True, exist_ok=True)

    seg_files = sorted(SEG_DIR.glob("*.seg"))
    if not seg_files:
        logger.error(f"No .seg files in {SEG_DIR}")
        sys.exit(1)

    logger.info(f"Found {len(seg_files)} .seg files")
    entries = []
    failed  = 0

    for seg_path in tqdm(seg_files, desc="Processing CFD"):
        img_path = IMAGES_DIR / (seg_path.stem + ".jpg")
        if not img_path.exists():
            img_path = IMAGES_DIR / (seg_path.stem + ".png")

        try:
            w, h = 480, 320
            if img_path.exists():
                with Image.open(img_path) as im:
                    w, h = im.size

            mask  = parse_seg_file(seg_path, w, h)
            boxes = mask_to_box(mask)

            mask_out = MASKS_DIR / (seg_path.stem + ".png")
            Image.fromarray(mask).save(mask_out)

            entries.append({
                "image":  str(img_path.resolve()),
                "mask":   str(mask_out.resolve()),
                "width":  w,
                "height": h,
                "boxes":  boxes,
            })
        except Exception as e:
            logger.warning(f"Failed {seg_path.name}: {e}")
            failed += 1

    logger.info(f"Processed {len(entries)} images ({failed} failed)")

    random.seed(seed)
    random.shuffle(entries)
    n_val = int(len(entries) * val_ratio)

    splits = {
        "sam_train": entries[n_val:],
        "sam_val":   entries[:n_val],
    }

    for name, data in splits.items():
        out_path = OUTPUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps(data, indent=2))
        logger.success(f"  [{name}] {len(data)} entries -> {out_path.name}")

    logger.success("CFD done. Masks in data/datasets/cfd/masks/")
    logger.info("Use sam_train.json with ml/segmentation/sam_inference.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()
    main(args.val_ratio, args.seed)