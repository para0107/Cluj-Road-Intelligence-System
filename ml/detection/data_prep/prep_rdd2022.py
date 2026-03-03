"""
ml/detection/data_prep/prep_rdd2022.py

PURPOSE:
    Converts RDD2022 from YOLO .txt format to COCO JSON.

USED FOR:
    Task  : Object detection training
    Model : RT-DETR-L
    Why   : Primary training dataset. ~38k images across Japan, India,
            USA, Norway, China, UAE. Covers all 4 main damage classes.

INPUT:
    data/datasets/rdd2022/
    ├── train/  ├── images/  └── labels/
    ├── val/    ├── images/  └── labels/
    └── test/   ├── images/  └── labels/

OUTPUT:
    data/datasets/rdd2022/
    ├── annotations_train.json
    ├── annotations_val.json
    └── annotations_test.json

CLASS MAPPING:
    RDD2022 0  D00  longitudinal_crack  -> unified 0
    RDD2022 1  D10  transverse_crack    -> unified 1
    RDD2022 2  D20  alligator_crack     -> unified 2
    RDD2022 3  D40  pothole             -> unified 3

USAGE:
    python ml/detection/data_prep/prep_rdd2022.py
"""

import sys
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger

ROOT        = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "data" / "datasets" / "rdd2022"

RDD2022_CLASS_MAP = {
    0: 0,  # longitudinal_crack
    1: 1,  # transverse_crack
    2: 2,  # alligator_crack
    3: 3,  # pothole
}

CATEGORIES = [
    {"id": 0, "name": "longitudinal_crack"},
    {"id": 1, "name": "transverse_crack"},
    {"id": 2, "name": "alligator_crack"},
    {"id": 3, "name": "pothole"},
    {"id": 4, "name": "patch_deterioration"},
]


def find_val_dir(dataset_dir):
    for name in ["val", "valid"]:
        d = dataset_dir / name
        if d.exists():
            return d
    return None


def yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h):
    abs_w = w  * img_w
    abs_h = h  * img_h
    abs_x = max(0.0, cx * img_w - abs_w / 2)
    abs_y = max(0.0,cy * img_h - abs_h / 2)
    return [abs_x, abs_y, abs_w, abs_h]


def build_coco(images_dir, labels_dir, split_name):
    coco = {
        "info":        {"description": f"RDD2022 {split_name} for RT-DETR-L"},
        "categories":  CATEGORIES,
        "images":      [],
        "annotations": [],
    }

    image_files    = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    ann_id         = 1
    missing_labels = 0
    skipped_anns   = 0

    for img_id, img_path in enumerate(tqdm(image_files, desc=f"  {split_name}"), start=1):
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        coco["images"].append({
            "id": img_id, "file_name": str(img_path.resolve()),
            "width": img_w, "height": img_h,
        })

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            missing_labels += 1
            continue

        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            rdd_cls = int(parts[0])
            if rdd_cls not in RDD2022_CLASS_MAP:
                skipped_anns += 1
                continue
            cx, cy, w, h = map(float, parts[1:])
            bbox = yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h)
            if bbox[2] <= 0 or bbox[3] <= 0:
                skipped_anns += 1
                continue
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": RDD2022_CLASS_MAP[rdd_cls],
                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
            })
            ann_id += 1

    logger.info(f"  missing labels: {missing_labels}  skipped: {skipped_anns}")
    return coco


def main(dataset_dir):
    logger.info("── RDD2022 -> COCO conversion ──────────────────────────────")
    if not dataset_dir.exists():
        logger.error(f"Not found: {dataset_dir}")
        sys.exit(1)

    splits = {
        "train": dataset_dir / "train",
        "val":   find_val_dir(dataset_dir),
        "test":  dataset_dir / "test",
    }

    for split_name, split_dir in splits.items():
        if split_dir is None or not split_dir.exists():
            logger.warning(f"Split '{split_name}' not found — skipping")
            continue

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"Missing images/ or labels/ in {split_dir} — skipping")
            continue

        logger.info(f"Processing: {split_name}/")
        coco     = build_coco(images_dir, labels_dir, split_name)
        out_path = dataset_dir / f"annotations_{split_name}.json"
        out_path.write_text(json.dumps(coco, indent=2))
        logger.success(
            f"  [{split_name}] {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations -> {out_path.name}"
        )

    logger.success("RDD2022 conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()
    main(args.dataset_dir)