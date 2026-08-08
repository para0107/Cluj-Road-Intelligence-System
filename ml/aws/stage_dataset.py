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
import os
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

# Hamming distance below which two perceptual hashes are treated as the same image.
# 5/64 bits is the conventional conservative threshold: tight enough not to collapse
# genuinely different road scenes, loose enough to catch re-encodes and adjacent
# video frames. Tune with --hash-threshold if the report shows false groupings.
HASH_THRESHOLD = 5

# Backwards-compatible alias. The old name is still referenced by scripts/ (which is
# gitignored and therefore not refactorable from here).
AHASH_THRESHOLD = HASH_THRESHOLD

# Default perceptual hash. See `dhash()` for why this is NOT aHash, and for why even
# dhash must be calibrated on the real archive before its threshold is trusted.
DEFAULT_HASH_ALGO = "dhash"

# Side length of the dHash comparison grid; the hash carries grid*grid bits.
HASH_GRID = 8

# Guard against the weekend-1 failure mode. aHash collapsed 72% of N-RDD2024 into
# false duplicate groups because an 8x8 greyscale average carries almost no signal on
# uniform grey asphalt, and union-find then chained everything into mega-groups. The
# number looked like a finding ("the dataset is 72% duplicates") rather than a bug.
#
# Near-duplicate merging above this fraction is treated as a broken hash rather than a
# dirty dataset, and staging aborts. Override with --max-neardup only after reading the
# printed example groups and confirming they really are the same photograph.
MAX_NEARDUP_FRACTION = 0.25


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass
class Sample:
    image: Path
    label: Optional[Path]
    sha256: str
    phash: Optional[int]       # perceptual hash; algorithm recorded in the manifest
    classes: tuple[int, ...]   # class ids present in this image

    @property
    def stem(self) -> str:
        return self.image.stem

    @property
    def ahash(self) -> Optional[int]:
        """Deprecated alias for `phash`, kept so gitignored scripts/ keep working."""
        return self.phash


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def _pixels(img) -> list[int]:
    """
    Flatten a greyscale PIL image to a list of ints.

    `Image.getdata()` is deprecated in Pillow 12 and removed in 14; `tobytes()` has
    been stable across every version this project has run on and needs no fallback
    branch. Mode "L" means one byte per pixel, so the byte order is the pixel order.
    """
    return list(img.tobytes())


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
    Average hash: downscale to 8x8 greyscale, threshold each pixel at the mean, pack
    into a 64-bit int.

    DO NOT USE THIS ON ROAD IMAGERY. Kept only so a re-stage can reproduce weekend 1's
    (wrong) numbers for comparison. aHash thresholds against the frame mean, so on a
    photograph that is mostly uniform grey asphalt every cell sits within noise of the
    mean and the output bits are close to arbitrary. Measured false-positive rate on
    N-RDD2024: 100%. Use `dhash`.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((8, 8), Image.Resampling.BILINEAR)
            px = _pixels(small)
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


def dhash(path: Path, grid: int = HASH_GRID) -> Optional[int]:
    """
    Difference hash: downscale to (grid+1) x grid greyscale and emit one bit per
    horizontally adjacent pair, giving grid*grid bits (64 at the default grid=8).

    Why this and not aHash. dHash encodes the SIGN OF THE LOCAL GRADIENT rather than
    each pixel's relation to the global mean. On uniform asphalt the gradient sign is
    still driven by real structure - lane markings, crack edges, tar seams, shadow
    boundaries - so the bits stay informative exactly where aHash degenerates.

    IMPORTANT, and not what EXPERIMENTS.md section 10 implies. dHash is strictly better
    than aHash here, but it is NOT automatically safe. Synthetic testing on smooth
    low-contrast dashcam-style frames collapsed dHash at grid=8 too (~90% false-positive
    pairs), because a bilinear downsample to 8x8 throws away exactly the mid-frequency
    detail that separates two frames of the same road. Raising the grid helps but trades
    against robustness to benign re-encoding, and no grid separated the two distributions
    cleanly on that fixture.

    The operational conclusion: do not trust ANY threshold you have not calibrated on
    the real archive. Run `--calibrate-hash` first, and if it does not report clean
    separation, stage with `--hash none` (exact duplicates only) and record the residual
    near-duplicate risk as a stated limitation, which is what weekend 1 effectively did.

    Returns None when the image cannot be read, which callers treat as "no near-duplicate
    information" rather than as a match.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    if grid < 2 or grid * grid > 1024:
        raise ValueError(f"grid must be in [2, 32], got {grid}")
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((grid + 1, grid), Image.Resampling.BILINEAR)
            px = _pixels(small)
    except Exception:
        return None
    if len(px) != (grid + 1) * grid:
        return None

    bits = 0
    k = 0
    for row in range(grid):
        base = row * (grid + 1)
        for col in range(grid):
            if px[base + col] > px[base + col + 1]:
                bits |= 1 << k
            k += 1
    return bits


# Registry so the algorithm is a recorded choice rather than a hardcoded call.
HASH_FUNCS: dict[str, object] = {"dhash": dhash, "ahash": ahash}


def perceptual_hash(path: Path, algo: str = DEFAULT_HASH_ALGO,
                    grid: int = HASH_GRID) -> Optional[int]:
    if algo == "none":
        return None
    if algo == "dhash":
        return dhash(path, grid)
    if algo == "ahash":
        return ahash(path)
    raise ValueError(f"unknown hash algo {algo!r}; expected one of "
                     f"{sorted(HASH_FUNCS)} or 'none'")


def calibrate_hash(
    source: Path, algo: str, grid: int, n_sample: int = 300, seed: int = 1337
) -> dict:
    """
    Decide whether a near-duplicate threshold is trustworthy ON THIS ARCHIVE, instead
    of inheriting a constant from a blog post.

    Method. Sample `n_sample` real images. For each, synthesise benign variants - the
    transforms a re-export actually applies (JPEG requantisation, a small rescale
    round-trip, a few-pixel crop) - and record the hash distance to its own variants.
    Separately record distances between DIFFERENT images. A threshold is usable only
    if those two distributions separate.

    Reported:
        recommended_threshold - the 95th percentile of the true-duplicate distances,
            i.e. the smallest threshold that still catches 95% of genuine re-encodes.
        false_positive_rate   - fraction of different-image pairs that threshold
            would wrongly merge. This is the number weekend 1 needed and did not have.
        separated             - True only when the false-positive rate is under 1%.

    A `separated: false` verdict is a real answer, not a failure: it means stage with
    `--hash none` and state the residual near-duplicate risk as a limitation.
    """
    import random as _random
    import statistics

    try:
        from PIL import Image
    except ImportError:
        return {"error": "Pillow not installed; cannot calibrate"}

    pairs = find_pairs(source)
    if len(pairs) < 20:
        return {"error": f"only {len(pairs)} images under {source}; need >= 20"}

    rng = _random.Random(seed)
    sample = rng.sample([p for p, _ in pairs], min(n_sample, len(pairs)))

    def variants(img_path: Path) -> Iterable[Path]:
        """Benign re-encodings, written to a temp dir."""
        import tempfile
        with Image.open(img_path) as im:
            im = im.convert("L")
            w, h = im.size
            tmpdir = Path(tempfile.mkdtemp())
            specs = [
                ("q60", im),
                ("scale", im.resize((int(w * 1.02), int(h * 1.02))).resize((w, h))),
                ("crop", im.crop((3, 3, w - 3, h - 3)).resize((w, h))),
            ]
            for tag, v in specs:
                out = tmpdir / f"{img_path.stem}_{tag}.jpg"
                v.save(out, quality=60)
                yield out

    true_d: list[int] = []
    for p in sample:
        h0 = perceptual_hash(p, algo, grid)
        if h0 is None:
            continue
        for v in variants(p):
            hv = perceptual_hash(v, algo, grid)
            if hv is not None:
                true_d.append(hamming(h0, hv))

    hashes = [h for h in (perceptual_hash(p, algo, grid) for p in sample) if h is not None]
    diff_d = [hamming(hashes[i], hashes[j])
              for i in range(len(hashes)) for j in range(i + 1, len(hashes))]

    if not true_d or not diff_d:
        return {"error": "could not hash enough images to calibrate"}

    true_sorted = sorted(true_d)
    thr = true_sorted[min(len(true_sorted) - 1, int(0.95 * len(true_sorted)))]
    fp_rate = sum(1 for d in diff_d if d <= thr) / len(diff_d)

    return {
        "algo": algo,
        "grid": grid if algo == "dhash" else None,
        "n_bits": grid * grid if algo == "dhash" else 64,
        "n_sampled_images": len(sample),
        "true_duplicate_distance": {
            "min": true_sorted[0], "median": int(statistics.median(true_sorted)),
            "p95": thr, "max": true_sorted[-1],
        },
        "different_image_distance": {
            "min": min(diff_d), "p5": sorted(diff_d)[int(0.05 * len(diff_d))],
            "median": int(statistics.median(diff_d)), "max": max(diff_d),
        },
        "recommended_threshold": thr,
        "false_positive_rate": round(fp_rate, 5),
        "separated": fp_rate < 0.01,
        "verdict": (
            f"usable: threshold {thr} catches 95% of re-encodes at "
            f"{fp_rate:.2%} false positives"
            if fp_rate < 0.01 else
            f"NOT usable: any threshold catching 95% of re-encodes also merges "
            f"{fp_rate:.1%} of genuinely different images. Stage with --hash none "
            f"(exact duplicates only) and state near-duplicate risk as a limitation."
        ),
    }


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _hash_bands(h: int, n_bands: int, band_bits: int) -> tuple[int, ...]:
    """
    Split a 64-bit hash into `n_bands` contiguous slices.

    Pigeonhole principle: two hashes differing in at most d bits must agree EXACTLY on
    at least one band whenever n_bands > d. Bucketing on bands therefore turns the
    O(n^2) all-pairs scan into a near-linear one with no false negatives - the
    candidate set is a strict superset of the true matches, and every candidate is
    still verified by an exact Hamming computation.
    """
    return tuple((h >> (i * band_bits)) & ((1 << band_bits) - 1) for i in range(n_bands))


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


def scan(root: Path, hash_algo: str = DEFAULT_HASH_ALGO,
         grid: int = HASH_GRID) -> list[Sample]:
    """Scan a YOLO-layout root into `Sample` records. `hash_algo='none'` skips hashing."""
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
            phash=perceptual_hash(img, hash_algo, grid),
            classes=read_classes(lab),
        ))
    return samples


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def group_duplicates(
    samples: list[Sample],
    threshold: int = HASH_THRESHOLD,
    max_neardup_fraction: float = MAX_NEARDUP_FRACTION,
    strict: bool = True,
) -> list[list[int]]:
    """
    Group samples that are the same image: exact SHA-256 matches first (free and
    certain), then near-duplicates by perceptual-hash Hamming distance within each
    SHA-distinct set.

    Returns a list of index groups; singletons included, so the groups partition the
    input. Every group must end up entirely inside ONE split - that is the invariant
    that prevents leakage.

    Args:
        threshold: maximum Hamming distance treated as the same photograph.
        max_neardup_fraction: abort if the near-duplicate pass absorbs more than this
            fraction of the SHA-distinct images. See `MAX_NEARDUP_FRACTION`.
        strict: when False, a suspected hash collapse warns instead of raising.

    Raises:
        RuntimeError: the near-duplicate pass merged implausibly much, which means the
            hash has degenerated on this imagery rather than the dataset being dirty.

    Complexity: the near-duplicate pass is banded (see `_hash_bands`), so it is close
    to linear rather than the previous O(n^2) all-pairs scan, with identical results.
    """
    by_sha: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_sha[s.sha256].append(i)

    # Representative of each exact-duplicate cluster.
    reps = [(idxs[0], idxs) for idxs in by_sha.values()]
    rep_ids = [r for r, _ in reps]
    n_sha_distinct = len(rep_ids)

    parent = {r: r for r in rep_ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Banded bucketing. n_bands must exceed the threshold for the pigeonhole argument
    # to hold; 64 bits / 6 bands = 10 bits each, covering thresholds up to 5.
    n_bands = max(threshold + 1, 6)
    band_bits = 64 // n_bands
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for r in rep_ids:
        h = samples[r].phash
        if h is None:
            continue
        for b, val in enumerate(_hash_bands(h, n_bands, band_bits)):
            buckets[(b, val)].append(r)

    n_merges = 0
    examples: list[tuple[int, int, int]] = []
    checked: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            ha = samples[members[a]].phash
            for b in range(a + 1, len(members)):
                key = (members[a], members[b])
                if key in checked:
                    continue
                checked.add(key)
                hb = samples[members[b]].phash
                d = hamming(ha, hb)  # type: ignore[arg-type]
                if d <= threshold:
                    if find(members[a]) != find(members[b]):
                        n_merges += 1
                        if len(examples) < 5:
                            examples.append((members[a], members[b], d))
                    union(members[a], members[b])

    # -- the guard that weekend 1 did not have ------------------------------
    frac = n_merges / max(n_sha_distinct, 1)
    if frac > max_neardup_fraction:
        detail = "\n".join(
            f"    d={d}  {samples[i].image}\n         {samples[j].image}"
            for i, j, d in examples
        )
        msg = (
            f"near-duplicate pass absorbed {n_merges}/{n_sha_distinct} "
            f"({frac:.1%}) of SHA-distinct images, above the "
            f"{max_neardup_fraction:.0%} ceiling.\n"
            f"This is the aHash collapse signature (EXPERIMENTS.md section 10): a "
            f"perceptual hash that has degenerated on low-contrast road imagery "
            f"reports the whole dataset as duplicates.\n"
            f"Inspect these pairs before believing the number:\n{detail}\n"
            f"If they really are the same photograph, re-run with "
            f"--max-neardup {min(1.0, frac + 0.05):.2f}."
        )
        if strict:
            raise RuntimeError(msg)
        print(f"[dedupe] WARNING: {msg}", file=sys.stderr)

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
    split_idx: dict[str, list[int]], samples: list[Sample], threshold: int = HASH_THRESHOLD
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
            if s.phash is None or s.sha256 in seen:
                continue
            seen[s.sha256] = i
            pairs.append((i, s.phash))
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
        "hash_threshold": threshold,
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
def _place(src: Path, dst: Path, mode: str) -> str:
    """
    Put `src` at `dst` by copy, hardlink or symlink. Returns the mode actually used.

    Why this is not just `shutil.copy2`. The E9/E10 programme stages SEVEN variants of
    the same ~19k images, and a full byte copy each time is seven times the disk for
    zero benefit - every variant reads the identical source files, and nothing in the
    pipeline ever writes to a staged image. On a 97 GB Studio volume that is the
    difference between comfortable and running out at hour 20.

    Falls back copy <- symlink <- hardlink rather than failing: a cross-filesystem
    hardlink raises OSError, and some filesystems disallow symlinks entirely. Losing
    disk efficiency is survivable; losing the run is not.

    `src` is resolved first because fetch_kaggle's merged view is ITSELF symlinks into
    the kagglehub cache, and a symlink pointing at a symlink breaks as soon as anything
    tidies the intermediate.
    """
    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"

    real = src.resolve()
    if mode == "hardlink":
        try:
            os.link(real, dst)
            return "hardlink"
        except OSError:
            mode = "symlink"        # different filesystem; fall through
    if mode == "symlink":
        try:
            os.symlink(real, dst)
            return "symlink"
        except OSError:
            pass
    shutil.copy2(real, dst)
    return "copy"


def materialise(
    split_idx: dict[str, list[int]], samples: list[Sample], out: Path, copy: bool = True,
    link_mode: str = "copy",
) -> dict:
    """
    Write the split to disk in YOLO layout. Duplicates get a __dupN suffix.

    Args:
        copy: False writes only the yaml and directory skeleton (used by --dry-run).
        link_mode: "copy", "symlink" or "hardlink". See `_place`.
    """
    out = Path(out)
    counts: dict[str, int] = {}
    modes_used: Counter = Counter()
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
            if copy and not (dst_img.exists() or dst_img.is_symlink()):
                modes_used[_place(s.image, dst_img, link_mode)] += 1
            if s.label and s.label.exists():
                dst_lab = lab_dir / f"{stem}.txt"
                # Labels are ~100 bytes and a class-set remap REWRITES them, so they
                # are always real copies. Symlinking a label would edit the source.
                if copy and not dst_lab.exists():
                    shutil.copy2(s.label, dst_lab)
        counts[split] = len(idxs)
    if modes_used:
        print(f"[write] images placed by: "
              f"{', '.join(f'{m}={n}' for m, n in modes_used.most_common())}")

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
    ap.add_argument("--hash", dest="hash_algo", default=DEFAULT_HASH_ALGO,
                    choices=["dhash", "ahash", "none"],
                    help="perceptual hash for near-duplicate detection. dhash is the "
                         "verified choice for road imagery; ahash is kept only to "
                         "reproduce weekend 1 and has a 100%% false-positive rate here")
    ap.add_argument("--hash-grid", type=int, default=HASH_GRID,
                    help="dHash comparison grid; the hash carries grid*grid bits")
    ap.add_argument("--hash-threshold", type=int, default=HASH_THRESHOLD)
    ap.add_argument("--calibrate-hash", type=int, nargs="?", const=300, metavar="N",
                    help="measure whether any near-duplicate threshold separates real "
                         "re-encodes from different images on THIS archive, then exit. "
                         "Run before trusting --hash-threshold")
    ap.add_argument("--ahash-threshold", type=int, default=None,
                    help=argparse.SUPPRESS)   # deprecated alias
    ap.add_argument("--max-neardup", type=float, default=MAX_NEARDUP_FRACTION,
                    help="abort if the near-duplicate pass absorbs more than this "
                         "fraction; guards against a collapsed hash")
    ap.add_argument("--oversample", type=float, default=0.30,
                    help="rare-class target as a fraction of the most common class")
    ap.add_argument("--no-oversample", action="store_true")
    ap.add_argument("--holdout-country", metavar="NAME",
                    help="leave-one-country-out: this country becomes the ENTIRE test "
                         "split (japan|usa|norway|china|india|czech). Overrides --ratios")
    ap.add_argument("--holdout-control", type=int, metavar="N",
                    help="the LOCO control: hold out N images at random instead of a "
                         "country, so the domain effect can be separated from the "
                         "train-size effect. Use the test size of the fold it controls")
    ap.add_argument("--link", dest="link_mode", default="symlink",
                    choices=["copy", "symlink", "hardlink"],
                    help="how to place images. symlink (default) makes the seven E9/E10 "
                         "variants nearly free on disk; copy is the safe fallback")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyse and report, copy nothing")
    args = ap.parse_args()

    if args.ahash_threshold is not None:
        args.hash_threshold = args.ahash_threshold
    if args.holdout_country and args.holdout_control:
        ap.error("--holdout-country and --holdout-control are alternatives; a control "
                 "run holds out random images INSTEAD of a country")

    if args.calibrate_hash is not None:
        if not args.source:
            ap.error("--calibrate-hash needs --source")
        rep = calibrate_hash(Path(args.source), args.hash_algo, args.hash_grid,
                             args.calibrate_hash, args.seed)
        print(json.dumps(rep, indent=2))
        if "error" in rep:
            return 1
        return 0 if rep["separated"] else 2

    if args.verify:
        root = Path(args.verify)
        print(f"[verify] scanning {root}")
        split_idx: dict[str, list[int]] = {}
        all_samples: list[Sample] = []
        for split in ("train", "val", "test"):
            d = root / split
            if not d.is_dir():
                continue
            s = scan(d, args.hash_algo, args.hash_grid)
            split_idx[split] = list(range(len(all_samples), len(all_samples) + len(s)))
            all_samples.extend(s)
        if not all_samples:
            print("[verify] nothing found", file=sys.stderr)
            return 1
        rep = verify_no_leakage(split_idx, all_samples, args.hash_threshold)
        print(json.dumps(rep, indent=2))
        return 0 if rep["clean"] else 2

    if not args.source or not args.out:
        ap.error("--source and --out are required unless --verify is used")

    source, out = Path(args.source), Path(args.out)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        ap.error(f"--ratios must be three numbers summing to 1.0, got {ratios}")

    print(f"[scan] {source}  (hash={args.hash_algo})")
    samples = scan(source, args.hash_algo, args.hash_grid)
    print(f"[scan] {len(samples)} images")

    print("[dedupe] grouping exact and near duplicates")
    try:
        groups = group_duplicates(samples, args.hash_threshold, args.max_neardup)
    except RuntimeError as e:
        print(f"[dedupe] ABORT: {e}", file=sys.stderr)
        return 3
    n_dupes = len(samples) - len(groups)
    print(f"[dedupe] {len(groups)} unique groups, {n_dupes} duplicate images absorbed")
    if n_dupes:
        print(f"[dedupe] NOTE: {n_dupes} duplicates existed in the SOURCE. If the "
              f"original train/val split was made without this grouping, the "
              f"published mAP is suspect.")

    # -- split ---------------------------------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml.research.geo_splits import (
        country_histogram, loco_split, matched_random_holdout,
    )

    split_mode = "random"
    if args.holdout_country or args.holdout_control:
        print("[geo] country composition of the source:")
        country_histogram(samples).report()

    if args.holdout_country:
        split_mode = f"loco:{args.holdout_country}"
        print(f"[split] LEAVE-ONE-COUNTRY-OUT, holdout={args.holdout_country}, "
              f"seed={args.seed}")
        try:
            split_idx = loco_split(groups, samples, args.holdout_country, seed=args.seed)
        except ValueError as e:
            print(f"[split] {e}", file=sys.stderr)
            return 3
        print("[split] REMINDER: a LOCO number alone conflates domain shift with the "
              "smaller training set. Pair it with --holdout-control "
              f"{len(split_idx['test'])}.")
    elif args.holdout_control:
        split_mode = f"control:{args.holdout_control}"
        print(f"[split] MATCHED RANDOM HOLDOUT (LOCO control), "
              f"target={args.holdout_control} images, seed={args.seed}")
        try:
            split_idx = matched_random_holdout(
                groups, samples, args.holdout_control, seed=args.seed)
        except ValueError as e:
            print(f"[split] {e}", file=sys.stderr)
            return 3
    else:
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
    leak = verify_no_leakage(split_idx, samples, args.hash_threshold)
    print(f"  exact cross-split duplicates: {leak['exact_duplicate_shas_across_splits']}")
    print(f"  near cross-split duplicates:  {leak['near_duplicate_pairs_across_splits']}")
    if not leak["clean"]:
        print("  LEAKAGE DETECTED - splits are not usable", file=sys.stderr)

    dist = class_distribution(split_idx, samples)

    manifest = {
        "source": str(source),
        "seed": args.seed,
        "split_mode": split_mode,
        "holdout_country": args.holdout_country,
        "holdout_control_target": args.holdout_control,
        "ratios": list(ratios) if split_mode == "random" else None,
        "link_mode": args.link_mode,
        "hash_algo": args.hash_algo,
        "hash_grid": args.hash_grid,
        "hash_threshold": args.hash_threshold,
        "max_neardup_fraction": args.max_neardup,
        "oversample_target_ratio": None if args.no_oversample else args.oversample,
        "n_source_images": len(samples),
        "n_unique_groups": len(groups),
        "n_source_duplicates": n_dupes,
        "split_sizes": {k: len(v) for k, v in split_idx.items()},
        "country_distribution": {
            split: dict(country_histogram([samples[i] for i in idxs]))
            for split, idxs in split_idx.items()
        },
        "class_distribution": dist,
        "leakage": leak,
    }

    if args.dry_run:
        print("\n[dry-run] nothing written. Manifest:\n")
        print(json.dumps(manifest, indent=2)[:4000])
        return 0 if leak["clean"] else 2

    print(f"[write] {out}")
    counts = materialise(split_idx, samples, out, copy=True,
                         link_mode=args.link_mode)
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
