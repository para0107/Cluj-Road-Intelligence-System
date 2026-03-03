"""
ml/detection/data_prep/merge_datasets.py

Merges RDD2022 and Pothole-600 COCO JSONs into unified train/val/test JSONs
for RT-DETR-L training using ID offsetting (no filesystem operations).

USAGE:
    python ml/detection/data_prep/merge_datasets.py
    python ml/detection/data_prep/merge_datasets.py --include rdd2022
"""

import json
import argparse
from pathlib import Path
from loguru import logger

ROOT       = Path(__file__).resolve().parents[3]
OUTPUT_DIR = ROOT / "data" / "detection"

CATEGORIES = [
    {"id": 0, "name": "longitudinal_crack",  "supercategory": "road_damage"},
    {"id": 1, "name": "transverse_crack",    "supercategory": "road_damage"},
    {"id": 2, "name": "alligator_crack",     "supercategory": "road_damage"},
    {"id": 3, "name": "pothole",             "supercategory": "road_damage"},
    {"id": 4, "name": "patch_deterioration", "supercategory": "road_damage"},
]

DATASETS = {
    "rdd2022": {
        "train": ROOT / "data" / "datasets" / "rdd2022" / "annotations_train.json",
        "val":   ROOT / "data" / "datasets" / "rdd2022" / "annotations_val.json",
        "test":  ROOT / "data" / "datasets" / "rdd2022" / "annotations_test.json",
    },
    "pothole600": {
        "train": ROOT / "data" / "datasets" / "pothole600" / "annotations_train.json",
        "val":   ROOT / "data" / "datasets" / "pothole600" / "annotations_val.json",
        "test":  ROOT / "data" / "datasets" / "pothole600" / "annotations_test.json",
    },
}


def load_coco(json_path: Path) -> dict:
    if not json_path.exists():
        logger.warning(f"  Not found: {json_path} — skipping")
        return {"images": [], "annotations": [], "categories": []}
    with open(json_path) as f:
        return json.load(f)


def merge_split(split: str, include: list) -> dict:
    merged = {
        "info": {
            "description": f"Merged road damage dataset — {split} split",
            "datasets":    include,
            "split":       split,
        },
        "categories":  CATEGORIES,
        "images":      [],
        "annotations": [],
    }

    img_id_offset = 0
    ann_id_offset = 0

    for ds_name in include:
        if ds_name not in DATASETS:
            logger.warning(f"Unknown dataset '{ds_name}' — skipping")
            continue

        json_path = DATASETS[ds_name].get(split)
        if json_path is None or not json_path.exists():
            logger.warning(f"  {ds_name} has no '{split}' split — skipping")
            continue

        coco = load_coco(json_path)
        if not coco["images"]:
            continue

        n_imgs = len(coco["images"])
        n_anns = len(coco["annotations"])
        logger.info(
            f"  [{split}] {ds_name}: {n_imgs:,} images, {n_anns:,} annotations"
            f"  (image ID offset: +{img_id_offset})"
        )

        for img in coco["images"]:
            merged["images"].append({
                "id":        img["id"] + img_id_offset,
                "file_name": img["file_name"],
                "width":     img.get("width",  0),
                "height":    img.get("height", 0),
                "source":    ds_name,
            })

        for ann in coco["annotations"]:
            merged["annotations"].append({
                "id":          ann["id"]       + ann_id_offset,
                "image_id":    ann["image_id"] + img_id_offset,
                "category_id": ann["category_id"],
                "bbox":        ann["bbox"],
                "area":        ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                "iscrowd":     ann.get("iscrowd", 0),
            })

        img_id_offset += n_imgs
        ann_id_offset += n_anns

    return merged


def main(include: list):
    logger.info("── Merging datasets → unified COCO JSON ────────────────────")
    logger.info(f"Datasets  : {include}")
    logger.info(f"Output dir: {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        logger.info(f"\nProcessing split: {split}")
        merged   = merge_split(split, include)
        out_path = OUTPUT_DIR / f"{split}.json"

        with open(out_path, "w") as f:
            json.dump(merged, f)

        logger.success(
            f"  [{split}] {len(merged['images']):,} images, "
            f"{len(merged['annotations']):,} annotations → {out_path.name}"
        )

    # Class distribution summary
    logger.info("\n── Class distribution in merged train.json ─────────────────")
    with open(OUTPUT_DIR / "train.json") as f:
        train = json.load(f)

    id_to_name   = {c["id"]: c["name"] for c in CATEGORIES}
    class_counts = {c["name"]: 0 for c in CATEGORIES}
    for ann in train["annotations"]:
        name = id_to_name.get(ann["category_id"], "unknown")
        class_counts[name] = class_counts.get(name, 0) + 1

    for cls_name, count in class_counts.items():
        logger.info(f"  {cls_name:<25} {count:>8,}")

    logger.success("\nMerge complete.")
    logger.success(f"  train : {OUTPUT_DIR / 'train.json'}")
    logger.success(f"  val   : {OUTPUT_DIR / 'val.json'}")
    logger.success(f"  test  : {OUTPUT_DIR / 'test.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include", nargs="+",
        default=["rdd2022", "pothole600"],
        choices=["rdd2022", "pothole600"],
    )
    args = parser.parse_args()
    main(args.include)