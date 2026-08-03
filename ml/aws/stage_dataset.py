"""
ml/aws/stage_dataset.py
-----------------------
Stage N-RDD2024 for the research program: build honest splits, prove there is no
leakage, and upload to S3 with a recorded fingerprint.

WHY THIS EXISTS
    The current training set is `train_oversampled/` and the yaml defines only
    `train` and `val`. Two problems follow.

    1. No test split. Every number the project holds was measured on the split that
       57 epochs of checkpoint selection ran against, so it is optimistically biased
       by an unknown amount.

    2. Oversampling before splitting is the classic way to leak. If a minority-class
       image was duplicated and the copies landed on both sides of the train/val
       boundary, the model was validated on images it trained on and mAP50 0.5637 is
       inflated. Nothing in the repo records which order was used.

    This script enforces the only safe order:

        dedupe originals -> split by ORIGINAL image -> oversample TRAIN ONLY -> verify

    Oversampling after the split cannot leak, because duplicates never cross a
    boundary that was drawn before they existed.

LEAKAGE DETECTION
    Exact duplicates are caught by SHA-256 of file bytes. Near-duplicates - re-encoded
    JPEGs, consecutive dashcam frames of the same damage - are caught by an average
    hash (aHash) over a 8x8 grayscale downscale, compared by Hamming distance.
    Near-duplicate frames are a real risk here: the source footage is video, and two
    frames 200 ms apart show the same pothole from almost the same angle. Splitting
    those across train and val is leakage even though the files differ.

    Reference: Zauner, 2010, "Implementation and Benchmarking of Perceptual Image
    Hash Functions" (aHash/pHash family).

USAGE
    # 1. Inspect what is there, split, verify - writes nothing to S3
    python ml/aws/stage_dataset.py --source /path/to/nrdd2024 \\
        --out /path/to/staged --dry-run

    # 2. Same, then upload
    python ml/aws/stage_dataset.py --source /path/to/nrdd2024 \\
        --out /path/to/staged --s3 s3://my-bucket/nrdd2024/v1

    # 3. Verify an existing staged directory without rebuilding it
    python ml/aws/stage_dataset.py --verify /path/to/staged

SOURCE DATA
    N-RDD2024, Kaya, O. & Codur, M. Y. (2024), doi:10.17632/27c8pwsd6v.3
    Download from Mendeley Data. This script does NOT download it for you - the
    Mendeley record requires accepting terms, and silently scraping it would be both
    rude and fragile. Point --source at the extracted directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# N-RDD2024 10-class schema. Mirrors pipeline/detector.py CLASS_NAMES.
CLASS_NAMES = [
    "longitudinal_crack", "transverse_crack", "alligator_crack", "repaired_crack",
    "pothole", "pedestrian_crossing_blur", "lane_line_blur", "manhole_cover",
    "patchy_road", "rutting",
]

# Hamming distance below which two aHashes are treated as the same image.
# 5/64 bits is the conventional conservative threshold: tight enough not to collapse
# genuinely different road scenes, loose enough to catch re-encodes and adjacent
# video frames. Tune with --ahash-threshold if the report shows false groupings.
AHASH_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    image: Path
    label: Optional[Path]
    sha256: str
    ahash: Optional[int]
    classes: tuple[int, ...]   # class ids present in this image

    @property
    def stem(self) -> str:
        return self.image.stem


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ahash(path: Path) -> Optional[int]:
    """
    Average hash: downscale to 8x8 grayscale, threshold each pixel at the mean,
    pack into a 64-bit int. Cheap and robust to re-encoding and mild rescaling.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
            px = list(small.getdata())
    except Exception:
        return None
    if not px:
        return None
    mean = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p >= mean:
            bits |= 1 << i
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def find_pairs(root: Path) -> list[tuple[Path, Optional[Path]]]:
    """
    Locate (image, label) pairs under a YOLO-layout root, however deeply nested.

    Handles the two conventions in the wild: a sibling `labels/` directory next to
    `images/`, and label .txt files sitting alongside the images.
    """
    pairs: list[tuple[Path, Optional[Path]]] = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS or not img.is_file():
            continue
        label = None
        # convention A: .../images/x.jpg -> .../labels/x.txt
        parts = list(img.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            cand = Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")
            if cand.exists():
                label = cand
        # convention B: sibling .txt
        if label is None:
            cand = img.with_suffix(".txt")
            if cand.exists():
                label = cand
        pairs.append((img, label))
    return pairs


def read_classes(label: Optional[Path]) -> tuple[int, ...]:
    if label is None or not label.exists():
        return ()
    out: list[int] = []
    for line in label.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 5:
            try:
                out.append(int(float(parts[0])))
            except ValueError:
                continue
    return tuple(sorted(set(out)))


def scan(root: Path, compute_ahash: bool = True) -> list[Sample]:
    pairs = find_pairs(root)
    if not pairs:
        raise RuntimeError(f"no images found under {root}")
    samples: list[Sample] = []
    for i, (img, lab) in enumerate(pairs, 1):
        if i % 1000 == 0:
            print(f"  scanned {i}/{len(pairs)}", file=sys.stderr)
        samples.append(Sample(
            image=img,
            label=lab,
            sha256=sha256_file(img),
            ahash=ahash(img) if compute_ahash else None,
            classes=read_classes(lab),
        ))
    return samples


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def group_duplicates(samples: list[Sample], threshold: int = AHASH_THRESHOLD) -> list[list[int]]:
    """
    Group samples that are the same image. Exact SHA matches first (free), then
    near-duplicates by aHash Hamming distance within each SHA-distinct set.

    Returns a list of index groups; singletons included, so the groups partition the
    input. Every group must end up entirely inside ONE split - that is the invariant
    that prevents leakage.

    Complexity is O(n^2) in the aHash pass. For N-RDD2024 (order 1e4 images) that is
    a few seconds. Bucket by the high bits first if this is ever run on 1e6 images.
    """
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_sha[s.sha256].append(i)

    # Representative of each exact-duplicate cluster.
    reps = [(idxs[0], idxs) for idxs in by_sha.values()]

    parent = {r: r for r, _ in reps}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rep_ids = [r for r, _ in reps]
    for i in range(len(rep_ids)):
        hi = samples[rep_ids[i]].ahash
        if hi is None:
            continue
        for j in range(i + 1, len(rep_ids)):
            hj = samples[rep_ids[j]].ahash
            if hj is None:
                continue
            if hamming(hi, hj) <= threshold:
                union(rep_ids[i], rep_ids[j])

    merged: dict[int, list[int]] = defaultdict(list)
    for rep, idxs in reps:
        merged[find(rep)].extend(idxs)
    return [sorted(v) for v in merged.values()]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_groups(
    groups: list[list[int]],
    samples: list[Sample],
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 1337,
) -> dict[str, list[int]]:
    """
    Assign whole duplicate-groups to train/val/test, stratified by the rarest class
    each group contains.

    Stratifying on the RAREST class is the right call for a long-tailed detection
    dataset: an image containing both a common longitudinal crack and a rare rutting
    instance matters far more to the rutting split than to the crack split. Balancing
    on the common class would let the rare class end up almost absent from test, and
    a per-class AP computed on four test instances is not a number worth reporting.
    """
    rng = random.Random(seed)

    class_freq = Counter()
    for s in samples:
        for c in s.classes:
            class_freq[c] += 1

    def stratum(group: list[int]) -> int:
        present: set[int] = set()
        for i in group:
            present.update(samples[i].classes)
        if not present:
            return -1  # background-only images
        return min(present, key=lambda c: class_freq.get(c, 0))

    by_stratum: dict[int, list[list[int]]] = defaultdict(list)
    for g in groups:
        by_stratum[stratum(g)].append(g)

    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    r_train, r_val, _ = ratios

    for _, gs in sorted(by_stratum.items()):
        gs = list(gs)
        rng.shuffle(gs)
        n = len(gs)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        # Guarantee test is non-empty for a stratum that has at least 3 groups,
        # otherwise a rare class can vanish from test entirely.
        if n >= 3 and n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        for k, g in enumerate(gs):
            split = "train" if k < n_train else ("val" if k < n_train + n_val else "test")
            out[split].extend(g)

    return out


def oversample_train(
    train_idx: list[int], samples: list[Sample], target_ratio: float = 0.30
) -> list[int]:
    """
    Duplicate TRAIN indices so rare classes reach at least `target_ratio` of the most
    common class's image count. Runs strictly after the split, so a duplicate can
    never cross into val or test.

    Returns the (longer) index list; duplicates are repeated indices, which the
    manifest writer materialises as distinct files with a `__dupN` suffix.
    """
    img_count = Counter()
    for i in train_idx:
        for c in samples[i].classes:
            img_count[c] += 1
    if not img_count:
        return list(train_idx)

    max_count = max(img_count.values())
    target = {c: int(max_count * target_ratio) for c in img_count}

    by_class: dict[int, list[int]] = defaultdict(list)
    for i in train_idx:
        for c in samples[i].classes:
            by_class[c].append(i)

    extra: list[int] = []
    rng = random.Random(1337)
    for c, have in img_count.items():
        need = target[c] - have
        if need > 0 and by_class[c]:
            extra.extend(rng.choices(by_class[c], k=need))
    return list(train_idx) + extra


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_no_leakage(
    split_idx: dict[str, list[int]], samples: list[Sample], threshold: int = AHASH_THRESHOLD
) -> dict:
    """
    Independent check that no image (exact or near-duplicate) appears in two splits.

    Deliberately re-derived from scratch rather than trusting the grouping that
    produced the split - a verification that reuses the logic it verifies proves
    nothing.
    """
    sha_to_splits: dict[str, set[str]] = defaultdict(set)
    for split, idxs in split_idx.items():
        for i in idxs:
            sha_to_splits[samples[i].sha256].add(split)
    exact = {sha: sorted(sp) for sha, sp in sha_to_splits.items() if len(sp) > 1}

    # Near-duplicate check across split boundaries only.
    reps: dict[str, list[tuple[int, int]]] = {}
    for split, idxs in split_idx.items():
        seen: dict[str, int] = {}
        pairs: list[tuple[int, int]] = []
        for i in idxs:
            s = samples[i]
            if s.ahash is None or s.sha256 in seen:
                continue
            seen[s.sha256] = i
            pairs.append((i, s.ahash))
        reps[split] = pairs

    near: list[dict] = []
    names = sorted(reps)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            for i, hi in reps[names[a]]:
                for j, hj in reps[names[b]]:
                    d = hamming(hi, hj)
                    if d <= threshold:
                        near.append({
                            "split_a": names[a], "image_a": str(samples[i].image),
                            "split_b": names[b], "image_b": str(samples[j].image),
                            "hamming": d,
                        })

    return {
        "exact_duplicate_shas_across_splits": len(exact),
        "exact_examples": dict(list(exact.items())[:10]),
        "near_duplicate_pairs_across_splits": len(near),
        "near_examples": near[:10],
        "clean": len(exact) == 0 and len(near) == 0,
        "ahash_threshold": threshold,
    }


def class_distribution(split_idx: dict[str, list[int]], samples: list[Sample]) -> dict:
    """Per-split, per-class INSTANCE counts (not image counts) for the report."""
    dist: dict[str, dict[str, int]] = {}
    for split, idxs in split_idx.items():
        c = Counter()
        for i in idxs:
            lab = samples[i].label
            if lab is None or not lab.exists():
                continue
            for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        cid = int(float(parts[0]))
                    except ValueError:
                        continue
                    name = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"class_{cid}"
                    c[name] += 1
        dist[split] = dict(sorted(c.items()))
    return dist


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------
def materialise(
    split_idx: dict[str, list[int]], samples: list[Sample], out: Path, copy: bool = True
) -> dict:
    """Write the split to disk in YOLO layout. Duplicates get a __dupN suffix."""
    out = Path(out)
    counts: dict[str, int] = {}
    for split, idxs in split_idx.items():
        img_dir, lab_dir = out / split / "images", out / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        seen: Counter = Counter()
        for i in idxs:
            s = samples[i]
            n = seen[i]
            seen[i] += 1
            stem = s.stem if n == 0 else f"{s.stem}__dup{n}"
            dst_img = img_dir / f"{stem}{s.image.suffix}"
            if copy:
                shutil.copy2(s.image, dst_img)
            if s.label and s.label.exists():
                dst_lab = lab_dir / f"{stem}.txt"
                if copy:
                    shutil.copy2(s.label, dst_lab)
        counts[split] = len(idxs)

    yaml = out / "dataset_nrdd2024_research.yaml"
    yaml.write_text(
        "# N-RDD2024 research splits - built by ml/aws/stage_dataset.py\n"
        "# Source: Kaya & Codur, doi:10.17632/27c8pwsd6v.3\n"
        "# Splits are group-disjoint: exact and near-duplicate images never cross a\n"
        "# split boundary. Oversampling was applied to TRAIN ONLY, after splitting.\n"
        f"path: {out}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    return counts


def upload_s3(local: Path, uri: str) -> None:
    """Sync the staged directory to S3. Requires boto3 and configured credentials."""
    try:
        import boto3
    except ImportError:
        raise RuntimeError("boto3 not installed: pip install boto3") from None

    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri}")
    bucket, _, prefix = uri[5:].partition("/")
    s3 = boto3.client("s3")

    files = [p for p in Path(local).rglob("*") if p.is_file()]
    print(f"[s3] uploading {len(files)} files to {uri}")
    for n, p in enumerate(files, 1):
        key = f"{prefix.rstrip('/')}/{p.relative_to(local).as_posix()}"
        s3.upload_file(str(p), bucket, key)
        if n % 500 == 0:
            print(f"  {n}/{len(files)}", file=sys.stderr)
    print(f"[s3] done: {uri}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Stage N-RDD2024 with honest splits")
    ap.add_argument("--source", help="extracted N-RDD2024 root")
    ap.add_argument("--out", help="where to write the staged splits")
    ap.add_argument("--verify", help="verify an already-staged dir, then exit")
    ap.add_argument("--s3", help="s3://bucket/prefix to upload to")
    ap.add_argument("--ratios", default="0.70,0.15,0.15")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--ahash-threshold", type=int, default=AHASH_THRESHOLD)
    ap.add_argument("--oversample", type=float, default=0.30,
                    help="rare-class target as a fraction of the most common class")
    ap.add_argument("--no-oversample", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse and report, copy nothing")
    args = ap.parse_args()

    if args.verify:
        root = Path(args.verify)
        print(f"[verify] scanning {root}")
        split_idx: dict[str, list[int]] = {}
        all_samples: list[Sample] = []
        for split in ("train", "val", "test"):
            d = root / split
            if not d.is_dir():
                continue
            s = scan(d)
            split_idx[split] = list(range(len(all_samples), len(all_samples) + len(s)))
            all_samples.extend(s)
        if not all_samples:
            print("[verify] nothing found", file=sys.stderr)
            return 1
        rep = verify_no_leakage(split_idx, all_samples, args.ahash_threshold)
        print(json.dumps(rep, indent=2))
        return 0 if rep["clean"] else 2

    if not args.source or not args.out:
        ap.error("--source and --out are required unless --verify is used")

    source, out = Path(args.source), Path(args.out)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        ap.error(f"--ratios must be three numbers summing to 1.0, got {ratios}")

    print(f"[scan] {source}")
    samples = scan(source)
    print(f"[scan] {len(samples)} images")

    print("[dedupe] grouping exact and near duplicates")
    groups = group_duplicates(samples, args.ahash_threshold)
    n_dupes = len(samples) - len(groups)
    print(f"[dedupe] {len(groups)} unique groups, {n_dupes} duplicate images absorbed")
    if n_dupes:
        print(f"[dedupe] NOTE: {n_dupes} duplicates existed in the SOURCE. If the "
              f"original train/val split was made without this grouping, the "
              f"published mAP is suspect.")

    print(f"[split] {ratios} stratified by rarest class, seed={args.seed}")
    split_idx = split_groups(groups, samples, ratios, args.seed)
    for k, v in split_idx.items():
        print(f"  {k:6s} {len(v):6d} images")

    if not args.no_oversample:
        before = len(split_idx["train"])
        split_idx["train"] = oversample_train(split_idx["train"], samples, args.oversample)
        print(f"[oversample] train {before} -> {len(split_idx['train'])} "
              f"(train only, after split - cannot leak)")

    print("[verify] independent leakage check")
    leak = verify_no_leakage(split_idx, samples, args.ahash_threshold)
    print(f"  exact cross-split duplicates: {leak['exact_duplicate_shas_across_splits']}")
    print(f"  near cross-split duplicates:  {leak['near_duplicate_pairs_across_splits']}")
    if not leak["clean"]:
        print("  LEAKAGE DETECTED - splits are not usable", file=sys.stderr)

    dist = class_distribution(split_idx, samples)

    manifest = {
        "source": str(source),
        "seed": args.seed,
        "ratios": list(ratios),
        "ahash_threshold": args.ahash_threshold,
        "oversample_target_ratio": None if args.no_oversample else args.oversample,
        "n_source_images": len(samples),
        "n_unique_groups": len(groups),
        "n_source_duplicates": n_dupes,
        "split_sizes": {k: len(v) for k, v in split_idx.items()},
        "class_distribution": dist,
        "leakage": leak,
    }

    if args.dry_run:
        print("\n[dry-run] nothing written. Manifest:\n")
        print(json.dumps(manifest, indent=2)[:4000])
        return 0 if leak["clean"] else 2

    print(f"[write] {out}")
    counts = materialise(split_idx, samples, out, copy=True)
    manifest["materialised_counts"] = counts

    from datetime import datetime, timezone
    manifest["staged_at"] = datetime.now(tz=timezone.utc).isoformat()

    # Fingerprint the result so a training run can prove which data it saw.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml.repro import hash_dataset
    manifest["dataset_hash"] = hash_dataset(out)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[write] manifest.json  sha256={manifest['dataset_hash']['sha256'][:16]}...")

    if args.s3:
        upload_s3(out, args.s3)

    return 0 if leak["clean"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
