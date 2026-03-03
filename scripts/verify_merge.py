import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "data" / "detection" / "train.json"

print(f"Loading {path} ...")
with open(path) as f:
    data = json.load(f)

imgs = data["images"]
anns = data["annotations"]

# 1. Duplicate image IDs
img_ids = [i["id"] for i in imgs]
assert len(img_ids) == len(set(img_ids)), "DUPLICATE IMAGE IDs"
print(f"[OK] Image IDs unique  — min={min(img_ids)}, max={max(img_ids)}, count={len(img_ids):,}")

# 2. Duplicate annotation IDs
ann_ids = [a["id"] for a in anns]
assert len(ann_ids) == len(set(ann_ids)), "DUPLICATE ANNOTATION IDs"
print(f"[OK] Ann IDs unique    — min={min(ann_ids)}, max={max(ann_ids)}, count={len(ann_ids):,}")

# 3. Orphaned annotations
valid_img_ids = set(img_ids)
orphans = [a for a in anns if a["image_id"] not in valid_img_ids]
assert len(orphans) == 0, f"{len(orphans)} ORPHANED ANNOTATIONS"
print(f"[OK] No orphaned annotations")

# 4. Non-overlapping ID ranges per source
for src in ["rdd2022", "pothole600"]:
    src_ids = [i["id"] for i in imgs if i["source"] == src]
    print(f"[OK] {src:<12} image IDs: {min(src_ids):>6} → {max(src_ids):<6}  count={len(src_ids):,}")

rdd_max  = max(i["id"] for i in imgs if i["source"] == "rdd2022")
p600_min = min(i["id"] for i in imgs if i["source"] == "pothole600")
assert rdd_max < p600_min,     "ID RANGES OVERLAP"
assert rdd_max + 1 == p600_min, "ID RANGES NOT CONTIGUOUS"
print(f"[OK] ID ranges non-overlapping and contiguous")

# 5. Valid bboxes
bad = [(a["id"], a["bbox"]) for a in anns
       if a["bbox"][0] < 0 or a["bbox"][1] < 0
       or a["bbox"][2] <= 0 or a["bbox"][3] <= 0]
if bad:
    print(f"[WARN] {len(bad)} invalid bboxes (negative or zero dimension)")
    for ann_id, bbox in bad[:5]:
        print(f"       ann_id={ann_id}  bbox={bbox}")
else:
    print(f"[OK] All bboxes valid (x≥0, y≥0, w>0, h>0)")

# 6. Category IDs all known
known_cats = {c["id"] for c in data["categories"]}
bad_cats   = [a for a in anns if a["category_id"] not in known_cats]
assert len(bad_cats) == 0, f"{len(bad_cats)} annotations with unknown category_id"
print(f"[OK] All category IDs valid — known categories: {sorted(known_cats)}")

print("\nAll checks passed ✓")