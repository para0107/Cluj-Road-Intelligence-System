"""
ml/detection/data_prep/prep_pothole600.py

PURPOSE:
    Converts Pothole-600 from PASCAL VOC XML → COCO JSON.

USED FOR:
    Task  : Object detection training
    Model : RT-DETR-L
    Why   : Supplements RDD2022 for class 3 (pothole) and class 4
            (patch_deterioration). Street-level/dashcam angle images
            closely match Cluj dashcam footage.

INPUT:
    data/datasets/pothole600/
    ├── images/       potholes0.png … potholes599.png
    └── annotations/  potholes0.xml … potholes599.xml

OUTPUT:
    data/datasets/pothole600/
    ├── annotations_train.json
    ├── annotations_val.json
    └── annotations_test.json

CLASS MAPPING (unified across all datasets):
    0  longitudinal_crack
    1  transverse_crack
    2  alligator_crack
    3  pothole             ← Pothole-600 "pothole"
    4  patch_deterioration

USAGE:
    python ml/detection/data_prep/prep_pothole600.py
    python ml/detection/data_prep/prep_pothole600.py --val_ratio 0.15 --test_ratio 0.15
"""

import os
import sys
import json
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parents[3]))

ROOT        = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT / "data" / "datasets" / "pothole600"
IMAGES_DIR  = DATASET_DIR / "images"
ANNOT_DIR   = DATASET_DIR / "annotations"

CLASS_MAP = {
    "pothole":             3,
    "patch_deterioration": 4,
}

CATEGORIES = [
    {"id": 0, "name": "longitudinal_crack"},
    {"id": 1, "name": "transverse_crack"},
    {"id": 2, "name": "alligator_crack"},
    {"id": 3, "name": "pothole"},
    {"id": 4, "name": "patch_deterioration"},
]


def parse_voc_xml(xml_path: Path):
    """
    Parse a single PASCAL VOC XML file.
    Returns (filename, width, height, list_of_boxes).
    Each box: {"label": str, "xmin": int, "ymin": int, "xmax": int, "ymax": int}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size     = root.find("size")
    width    = int(size.findtext("width"))
    height   = int(size.findtext("height"))

    boxes = []
    for obj in root.findall("object"):
        label = obj.findtext("name").strip().lower()
        bb    = obj.find("bndbox")
        boxes.append({
            "label": label,
            "xmin":  int(float(bb.findtext("xmin"))),
            "ymin":  int(float(bb.findtext("ymin"))),
            "xmax":  int(float(bb.findtext("xmax"))),
            "ymax":  int(float(bb.findtext("ymax"))),
        })
    return filename, width, height, boxes


def build_coco(samples):
    """
    Build a COCO-format dict from a list of
    (image_id, xml_path, image_path) tuples.
    """
    coco = {
        "info":        {"description": "Pothole-600 converted to COCO for RT-DETR-L training"},
        "categories":  CATEGORIES,
        "images":      [],
        "annotations": [],
    }

    ann_id         = 1
    skipped_labels = set()

    for img_id, xml_path, img_path in tqdm(samples, desc="Converting"):
        _, width, height, boxes = parse_voc_xml(xml_path)

        # Trust actual image dimensions over XML metadata
        if img_path.exists():
            with Image.open(img_path) as im:
                width, height = im.size

        coco["images"].append({
            "id": img_id,
            "file_name": str(img_path.resolve()),
            "width": width,
            "height": height,
        })
        for box in boxes:
            label = box["label"]
            if label not in CLASS_MAP:
                skipped_labels.add(label)
                continue

            x = box["xmin"]
            y = box["ymin"]
            w = box["xmax"] - box["xmin"]
            h = box["ymax"] - box["ymin"]

            if w <= 0 or h <= 0:
                continue

            coco["annotations"].append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": CLASS_MAP[label],
                "bbox":        [x, y, w, h],   # COCO: [x, y, width, height]
                "area":        w * h,
                "iscrowd":     0,
            })
            ann_id += 1

    if skipped_labels:
        logger.warning(f"Skipped unknown labels: {skipped_labels}")

    return coco


def main(val_ratio=0.15, test_ratio=0.15, seed=42):
    logger.info("── Pothole-600 → COCO conversion ──────────────────────────")
    logger.info(f"Dataset dir : {DATASET_DIR}")

    xml_files = sorted(ANNOT_DIR.glob("*.xml"))
    if not xml_files:
        logger.error(f"No XML files found in {ANNOT_DIR}")
        sys.exit(1)

    samples        = []
    missing_images = 0
    for xml_path in xml_files:
        img_path = IMAGES_DIR / (xml_path.stem + ".png")
        if not img_path.exists():
            img_path = IMAGES_DIR / (xml_path.stem + ".jpg")
        if not img_path.exists():
            missing_images += 1
            continue
        samples.append((xml_path, img_path))

    logger.info(f"Found {len(samples)} image/annotation pairs "
                f"({missing_images} missing images skipped)")

    random.seed(seed)
    random.shuffle(samples)

    n       = len(samples)
    n_test  = int(n * test_ratio)
    n_val   = int(n * val_ratio)
    n_train = n - n_val - n_test

    splits = {
        "train": samples[:n_train],
        "val":   samples[n_train:n_train + n_val],
        "test":  samples[n_train + n_val:],
    }

    logger.info(f"Split → train={n_train}  val={n_val}  test={n_test}")

    for split_name, split_samples in splits.items():
        indexed  = [(i + 1, xml, img) for i, (xml, img) in enumerate(split_samples)]
        coco     = build_coco(indexed)
        out_path = DATASET_DIR / f"annotations_{split_name}.json"

        with open(out_path, "w") as f:
            json.dump(coco, f, indent=2)

        logger.success(
            f"  [{split_name}] {len(coco['images'])} images, "
            f"{len(coco['annotations'])} annotations → {out_path.name}"
        )

    logger.success("Pothole-600 conversion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_ratio",  type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()
    main(args.val_ratio, args.test_ratio, args.seed)