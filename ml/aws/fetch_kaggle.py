"""
ml/aws/fetch_kaggle.py
----------------------
Download N-RDD2024 and RDD2022 from Kaggle, work out what layout they actually
have, and verify the class ordering before anything downstream trusts it.

THE CHECK THAT MATTERS

    A community Kaggle mirror is not guaranteed to use the same class ORDER as the
    original release. If the mirror's class 4 is `rutting` where the canonical schema
    has `pothole`, every label is silently wrong: training runs, metrics print, per-
    class AP looks plausible, and the entire result is meaningless. Nothing crashes.

    So this script does not just download. It reads whatever `data.yaml` /
    `classes.txt` ships with the dataset, compares the names AND their order against
    ml/research/class_sets.py, and refuses to hand off to staging when they disagree.

    If there is no class list at all, it reports the class-id histogram so you can
    reason about it yourself rather than assume.

WHAT ELSE IT HANDLES

    - Detects YOLO / COCO / PASCAL VOC annotations rather than assuming YOLO
    - Finds the real dataset root inside the arbitrary nesting Kaggle archives use
    - Reports any existing train/valid/test split, and explains why we re-split anyway
    - Warns on disk space before a multi-GB download rather than after

USAGE

    # Everything: download, verify, stage, upload
    python ml/aws/fetch_kaggle.py --all --s3 s3://<bucket>/nrdd2024/v1

    # Just look at what is there, change nothing
    python ml/aws/fetch_kaggle.py --dataset nrdd2024 --inspect-only

    # Download + verify + stage locally, no upload
    python ml/aws/fetch_kaggle.py --dataset nrdd2024 --stage /tmp/staged

AUTH
    kagglehub reads ~/.kaggle/kaggle.json, or KAGGLE_USERNAME + KAGGLE_KEY.
    Get the token from Kaggle -> Settings -> API -> Create New Token.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.research.class_sets import NRDD2024_CLASSES  # noqa: E402
from ml.research.datasets import get_dataset  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The Kaggle mirrors the user located.
KAGGLE_SOURCES: dict[str, dict] = {
    "nrdd2024": {
        "kaggle_id": "sannyshankaranml/n-rdd2024",
        "expected_classes": NRDD2024_CLASSES,
        "expected_nc": 10,
    },
    "rdd2022": {
        "kaggle_id": "aliabdelmenam/rdd-2022",
        "expected_classes": get_dataset("rdd2022").classes,
        "expected_nc": 4,
    },
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def check_disk(path: Path, need_gb: int = 25) -> bool:
    try:
        free_gb = shutil.disk_usage(path).free / 1024**3
    except OSError:
        return True
    if free_gb < need_gb:
        print(f"[disk] only {free_gb:.0f} GB free, want >= {need_gb} GB.\n"
              f"       kagglehub caches under ~/.cache/kagglehub, and staging needs "
              f"roughly another copy on top.", file=sys.stderr)
        return False
    print(f"[disk] {free_gb:.0f} GB free")
    return True


def download(kaggle_id: str) -> Path:
    """Download via kagglehub, falling back to the kaggle CLI."""
    try:
        import kagglehub
    except ImportError:
        print("[kaggle] installing kagglehub")
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "kagglehub"],
                       check=False)
        try:
            import kagglehub
        except ImportError:
            kagglehub = None  # type: ignore

    if kagglehub is not None:
        try:
            print(f"[kaggle] downloading {kaggle_id} (this is the slow part)")
            p = Path(kagglehub.dataset_download(kaggle_id))
            print(f"[kaggle] cached at {p}")
            return p
        except Exception as exc:
            print(f"[kaggle] kagglehub failed: {exc}", file=sys.stderr)

    # Fallback: the CLI, which sometimes authenticates where kagglehub does not.
    dest = Path("/tmp") / kaggle_id.replace("/", "_")
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] trying the kaggle CLI -> {dest}")
    r = subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(dest), "--unzip"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"both kagglehub and the kaggle CLI failed.\n{r.stderr[:600]}\n\n"
            "Auth: put kaggle.json in ~/.kaggle/ (chmod 600), or export "
            "KAGGLE_USERNAME and KAGGLE_KEY."
        )
    return dest


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------
def inspect(root: Path) -> dict:
    """
    Walk the download and report what is actually in it.

    Kaggle archives nest arbitrarily - the real dataset root is often two or three
    directories down, sometimes duplicated. Guessing the layout is how a staging run
    silently produces an empty dataset, so it is measured instead.
    """
    root = Path(root)
    images: list[Path] = []
    yolo_labels: list[Path] = []
    coco_jsons: list[Path] = []
    voc_xmls: list[Path] = []
    yamls: list[Path] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        sfx = p.suffix.lower()
        if sfx in IMAGE_EXTS:
            images.append(p)
        elif sfx == ".txt":
            # A YOLO label looks like "<int> <float> <float> <float> <float>".
            # README.txt and classes.txt must not be counted as annotations.
            try:
                first = p.read_text(encoding="utf-8", errors="replace").strip().split("\n")[0]
            except OSError:
                continue
            parts = first.split()
            if len(parts) == 5:
                try:
                    int(float(parts[0]))
                    [float(x) for x in parts[1:]]
                    yolo_labels.append(p)
                except ValueError:
                    pass
        elif sfx == ".json" and p.stat().st_size > 1000:
            coco_jsons.append(p)
        elif sfx == ".xml":
            voc_xmls.append(p)
        elif sfx in (".yaml", ".yml"):
            yamls.append(p)

    if yolo_labels:
        fmt = "yolo"
    elif voc_xmls:
        fmt = "voc"
    elif coco_jsons:
        fmt = "coco"
    else:
        fmt = "unknown"

    # Existing split directories, if any.
    splits = sorted({
        part.lower()
        for p in images[:4000]
        for part in p.parts
        if part.lower() in ("train", "val", "valid", "validation", "test")
    })

    # The directory that holds most images is the real root.
    parents = Counter(p.parent for p in images)
    dominant = parents.most_common(1)[0][0] if parents else root

    yolo_roots = find_yolo_roots(images, yolo_labels)

    return {
        "root": str(root),
        "format": fmt,
        "yolo_roots": yolo_roots,
        "n_images": len(images),
        "n_yolo_labels": len(yolo_labels),
        "n_voc_xml": len(voc_xmls),
        "n_coco_json": len(coco_jsons),
        "yaml_files": [str(y) for y in yamls[:5]],
        "existing_splits": splits,
        "dominant_image_dir": str(dominant),
        "_images": images,
        "_yolo_labels": yolo_labels,
        "_yamls": yamls,
    }


def find_yolo_roots(images: list[Path], yolo_labels: list[Path],
                    top: int = 40) -> list[dict]:
    """
    Find the subtree(s) that actually contain MATCHED image/label pairs.

    Necessary because research archives ship the same photographs several times over:
    once per annotation format (coco / voc / yolo) and once per country. Picking "the
    directory with the most images" lands in a COCO folder that has no YOLO labels at
    all, and staging then silently produces an empty dataset.

    Matching is by file STEM, which is the only thing YOLO actually relies on: an
    image at .../images/x.jpg is paired with .../labels/x.txt.

    Returns candidate roots sorted by matched-pair count, so the caller can choose
    the real one and the user can see the alternatives that were rejected.
    """
    label_stems: dict[Path, set[str]] = {}
    for lp in yolo_labels:
        label_stems.setdefault(lp.parent, set()).add(lp.stem)

    image_stems: dict[Path, set[str]] = {}
    for ip in images:
        image_stems.setdefault(ip.parent, set()).add(ip.stem)

    candidates: list[dict] = []
    for img_dir, stems in image_stems.items():
        best_dir, best_n = None, 0
        # ONLY the sibling `labels/` dir, or labels alongside the images. Matching
        # against any label directory that happens to share stems is too permissive:
        # these archives ship <country>_coco/, <country>_voc/ and <country>_txt/ with
        # IDENTICAL filenames, so a loose match pairs the COCO copy of a photo with
        # the YOLO labels and every image gets pooled two or three times over.
        for cand in (img_dir.parent / "labels", img_dir):
            if cand not in label_stems:
                continue
            n = len(stems & label_stems[cand])
            if n > best_n:
                best_dir, best_n = cand, n
        if best_n:
            candidates.append({
                "images_dir": str(img_dir),
                "labels_dir": str(best_dir),
                "n_images": len(stems),
                "n_matched": best_n,
                # The root stage_dataset.py should be pointed at: the parent holding
                # both images/ and labels/.
                "stage_root": str(img_dir.parent),
            })

    candidates.sort(key=lambda c: -c["n_matched"])
    return candidates[:top]


def build_merged_view(roots: list[dict], dest: Path) -> dict:
    """
    Pool every matched image/label subtree into one images/ + labels/ pair.

    Needed because these archives are organised by COUNTRY and by FORMAT:

        Training and Validation Dataset/
            japan_txt/train/{images,labels}      <- YOLO, usable
            japan_txt/valid/{images,labels}      <- YOLO, usable
            japan_coco/...                       <- same photos, COCO
            japan_voc/...                        <- same photos, VOC

    Pointing the stager at any single subtree throws away five sixths of the data;
    pointing it at the common parent sweeps in the coco/voc copies as label-less
    images, which YOLO would then train on as empty background frames.

    So the matched subtrees are symlinked into one flat view. Filenames are prefixed
    with their subtree tag, because two countries can legitimately contain the same
    stem and a silent overwrite would drop images without telling you.

    The unlabelled "Test Dataset" is excluded automatically - it has no labels, so it
    produces no matched pairs. That is correct: the official test annotations are
    withheld, which is exactly why our own test split has to be carved from the
    labelled data.
    """
    img_dir, lab_dir = dest / "images", dest / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    linked = skipped = 0
    per_subtree: dict[str, int] = {}

    for r in roots:
        src_img, src_lab = Path(r["images_dir"]), Path(r["labels_dir"])
        # Tag from the last two path components, e.g. "japan_txt_train".
        parts = [p for p in Path(r["stage_root"]).parts[-2:]]
        tag = "_".join(parts).replace(" ", "-").replace(".", "")
        n_here = 0

        for ip in sorted(src_img.iterdir()):
            if ip.suffix.lower() not in IMAGE_EXTS:
                continue
            lp = src_lab / f"{ip.stem}.txt"
            if not lp.exists():
                skipped += 1
                continue
            stem = f"{tag}__{ip.stem}"
            di, dl = img_dir / f"{stem}{ip.suffix}", lab_dir / f"{stem}.txt"
            for src, dst in ((ip, di), (lp, dl)):
                if dst.exists() or dst.is_symlink():
                    continue
                try:
                    os.symlink(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            linked += 1
            n_here += 1
        per_subtree[tag] = n_here

    return {"dest": str(dest), "linked_pairs": linked,
            "skipped_unlabelled": skipped, "per_subtree": per_subtree}


def read_declared_classes(yamls: list[Path], root: Path) -> Optional[list[str]]:
    """
    Pull the class list out of a data.yaml, or a classes.txt / *.names file.

    Parsed by hand rather than with PyYAML: the `names:` block appears both as a
    mapping (`0: pothole`) and as a list (`- pothole`), and a 20-line parser here is
    more predictable than a dependency that may not be installed yet.
    """
    for y in yamls:
        try:
            text = y.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "names" not in text:
            continue

        names: list[str] = []
        mapping: dict[int, str] = {}
        in_names = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("names"):
                in_names = True
                # inline list form: names: [a, b, c]
                if "[" in stripped:
                    inner = stripped[stripped.index("[") + 1: stripped.rindex("]")]
                    names = [s.strip().strip("'\"") for s in inner.split(",") if s.strip()]
                    break
                continue
            if in_names:
                if stripped.startswith("- "):
                    names.append(stripped[2:].strip().strip("'\""))
                elif ":" in stripped and stripped[0].isdigit():
                    k, _, v = stripped.partition(":")
                    try:
                        mapping[int(k.strip())] = v.strip().strip("'\"")
                    except ValueError:
                        pass
                elif stripped and not stripped.startswith("#"):
                    break
        if mapping:
            return [mapping[i] for i in sorted(mapping)]
        if names:
            return names

    for cand in ("classes.txt", "obj.names", "classes.names"):
        for p in root.rglob(cand):
            try:
                lines = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
            except OSError:
                continue
            if lines:
                print(f"[classes] read from {p}")
                return lines
    return None


def class_histogram(labels: list[Path], limit: int = 3000) -> Counter:
    """Class-id frequency, so a schema mismatch is visible even with no names file."""
    c: Counter = Counter()
    for p in labels[:limit]:
        try:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        c[int(float(parts[0]))] += 1
                    except ValueError:
                        pass
        except OSError:
            continue
    return c


def _canonicalise(names: list[str], expected: list[str]) -> tuple[list[str], bool]:
    """
    Translate Japanese D-codes to canonical English names.

    The N-RDD2024 and RDD2022 releases label classes as D00, D10, D20 ... D90. Those
    are the SAME categories as the English names, and ml/research/datasets.py already
    carries the mapping. Comparing "d00" against "longitudinal_crack" as raw strings
    reports a schema mismatch on a dataset that is in fact perfectly correct - which
    is a false alarm that costs an hour of doubt at exactly the wrong moment.

    Returns (translated_names, was_translated).
    """
    from ml.research.datasets import DATASETS

    code_to_name: dict[str, str] = {}
    for ds in DATASETS.values():
        for name, code in ds.d_codes.items():
            code_to_name[code.lower()] = name.lower()

    if not all(n.lower() in code_to_name for n in names):
        return names, False
    return [code_to_name[n.lower()] for n in names], True


def verify_schema(declared: Optional[list[str]], expected: list[str],
                  hist: Counter) -> tuple[bool, list[str]]:
    """
    Compare the dataset's class ordering with the canonical schema.

    Returns (safe_to_proceed, messages). Order matters as much as membership: the
    same ten names in a different order is a silent relabelling of every box.
    """
    msgs: list[str] = []
    ids = sorted(hist)
    max_id = max(ids) if ids else -1

    if max_id >= len(expected):
        msgs.append(
            f"FAIL class id {max_id} appears in the labels but the expected schema "
            f"has only {len(expected)} classes (0-{len(expected)-1}). This dataset "
            f"does not match the schema."
        )
        return False, msgs

    if declared is None:
        msgs.append(
            "WARN no class-name list found (no data.yaml, classes.txt or *.names). "
            "Class ORDER cannot be verified automatically."
        )
        msgs.append(f"     ids present: {ids}")
        msgs.append(f"     assuming: {list(enumerate(expected))}")
        msgs.append(
            "     Open a few images against their labels and confirm before trusting "
            "any result. A wrong order trains fine and reports nonsense."
        )
        return True, msgs

    norm = [n.strip().lower().replace(" ", "_").replace("-", "_") for n in declared]
    exp = [n.lower() for n in expected]

    norm, translated = _canonicalise(norm, exp)
    if translated:
        msgs.append("note dataset labels classes by Japanese D-code (D00..D90); "
                    "translated to canonical names via ml/research/datasets.py")

    if norm == exp:
        msgs.append(f"ok   class names and ORDER match the canonical schema "
                    f"({len(exp)} classes)")
        return True, msgs

    if sorted(norm) == sorted(exp):
        msgs.append("FAIL same class names but a DIFFERENT ORDER.")
        msgs.append(f"     dataset:  {norm}")
        msgs.append(f"     expected: {exp}")
        msgs.append(
            "     Every label id means something different here. Do NOT train on "
            "this until the ids are remapped - training would succeed and every "
            "number would be wrong."
        )
        return False, msgs

    msgs.append("FAIL class names differ from the canonical schema.")
    msgs.append(f"     dataset ({len(norm)}):  {norm}")
    msgs.append(f"     expected ({len(exp)}): {exp}")
    only_d = [n for n in norm if n not in exp]
    only_e = [n for n in exp if n not in norm]
    if only_d:
        msgs.append(f"     only in dataset:  {only_d}")
    if only_e:
        msgs.append(f"     missing:          {only_e}")
    return False, msgs


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def report(name: str, info: dict, declared: Optional[list[str]],
           hist: Counter, expected: list[str]) -> bool:
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    print(f"  root            {info['root']}")
    print(f"  format          {info['format']}")
    print(f"  images          {info['n_images']:,}")
    print(f"  yolo labels     {info['n_yolo_labels']:,}")
    if info["n_voc_xml"]:
        print(f"  voc xml         {info['n_voc_xml']:,}")
    if info["n_coco_json"]:
        print(f"  coco json       {info['n_coco_json']}")
    print(f"  existing splits {info['existing_splits'] or '(none)'}")

    roots = info.get("yolo_roots") or []
    if roots:
        print(f"\n  matched image/label subtrees ({len(roots)} found):")
        for i, r in enumerate(roots):
            mark = ""
            short = r["stage_root"]
            if len(short) > 78:
                short = "..." + short[-75:]
            print(f"    {r['n_matched']:6,} pairs  {short}{mark}")
        total_pairs = sum(r["n_matched"] for r in roots)
        print(f"    {total_pairs:6,} pairs  TOTAL - all of these are pooled")
        if info["n_images"] > total_pairs * 1.5:
            print(f"\n  NOTE {info['n_images']:,} images but only {total_pairs:,} "
                  f"matched pairs across these subtrees.\n"
                  f"       Research archives commonly ship the same photographs once "
                  f"per annotation format\n       (coco / voc / yolo) and once per "
                  f"country. Only the matched subtree is usable.")
    else:
        print(f"  main image dir  {info['dominant_image_dir']}")
        print("\n  NO matched image/label pairs found. The YOLO labels do not share "
              "stems with any\n  image directory, so this archive needs conversion "
              "before staging.")

    if info["format"] != "yolo":
        print(f"\n  NOTE format is '{info['format']}', not YOLO. stage_dataset.py "
              f"expects YOLO.")
        if info["format"] == "voc":
            print("       Convert first: ml/detection/data_prep/prep_rdd2022.py")
        elif info["format"] == "coco":
            print("       Convert first: ml/detection/data_prep/coco_to_yolo.py")
        return False

    if hist:
        total = sum(hist.values())
        print(f"\n  class distribution (sampled, {total:,} boxes):")
        for cid in sorted(hist):
            nm = expected[cid] if cid < len(expected) else "?"
            pct = 100.0 * hist[cid] / total
            bar = "#" * max(1, int(pct / 2))
            print(f"    {cid:2d} {nm:26s} {hist[cid]:7,} {pct:5.1f}%  {bar}")

    ok, msgs = verify_schema(declared, expected, hist)
    print()
    for m in msgs:
        print("  " + m)

    if info["existing_splits"]:
        print(f"\n  This dataset already has {info['existing_splits']} splits. We "
              f"re-split anyway, because\n  the existing boundary carries no proof "
              f"that duplicates were kept on one side of it,\n  and a leak there "
              f"inflates every number measured against it.")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def process(name: str, stage_to: Optional[str], s3: Optional[str],
            inspect_only: bool, force: bool,
            source_subdir: Optional[str] = None) -> int:
    src = KAGGLE_SOURCES[name]
    print(f"\n### {name}  ({src['kaggle_id']})")

    if not check_disk(Path.home()) and not force:
        print("[abort] not enough disk. Raise the space's EBS volume, or pass --force.",
              file=sys.stderr)
        return 1

    path = download(src["kaggle_id"])
    info = inspect(path)
    declared = read_declared_classes(info["_yamls"], path)
    hist = class_histogram(info["_yolo_labels"])
    ok = report(name, info, declared, hist, src["expected_classes"])

    out = {k: v for k, v in info.items() if not k.startswith("_")}
    out.update({"declared_classes": declared,
                "class_histogram": {str(k): v for k, v in sorted(hist.items())},
                "schema_ok": ok, "kaggle_id": src["kaggle_id"]})
    rep = Path("runs/research") / f"_kaggle_{name}.json"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  inspection written to {rep}")

    if inspect_only:
        return 0
    if not ok and not force:
        print("\n[abort] schema verification failed. Staging would produce a dataset "
              "whose labels do not mean what the code thinks they mean.\n"
              "        Fix the ordering, or re-run with --force if you have checked "
              "it yourself.", file=sys.stderr)
        return 2
    if not stage_to and not s3:
        print("\n  (no --stage or --s3 given, stopping after inspection)")
        return 0

    stage_dir = stage_to or f"/tmp/staged_{name}"
    roots = info.get("yolo_roots") or []
    if not roots and not source_subdir:
        print("\n[abort] no matched image/label pairs - nothing to stage.",
              file=sys.stderr)
        return 2

    if source_subdir:
        source = source_subdir
        print(f"\n[stage] source (manual): {source}")
    else:
        merged = Path("/tmp") / f"merged_{name}"
        print(f"\n[merge] pooling {len(roots)} matched subtree(s) into {merged}")
        m = build_merged_view(roots, merged)
        for tag, n in sorted(m["per_subtree"].items(), key=lambda kv: -kv[1]):
            print(f"    {n:7,}  {tag}")
        print(f"[merge] {m['linked_pairs']:,} image/label pairs pooled"
              + (f", {m['skipped_unlabelled']:,} unlabelled images skipped"
                 if m["skipped_unlabelled"] else ""))
        source = str(merged)

    cmd = [sys.executable, str(ROOT / "ml/aws/stage_dataset.py"),
           "--source", source, "--out", stage_dir]
    if s3:
        cmd += ["--s3", s3]

    print(f"\n[stage] {' '.join(cmd)}")
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        print(f"\n[done] staged to {stage_dir}")
        print(f"       yaml: {stage_dir}/dataset_nrdd2024_research.yaml")
    elif rc == 2:
        print("\n[done] staged, but LEAKAGE was detected. Read the report before "
              "training on it.", file=sys.stderr)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and verify the Kaggle datasets")
    ap.add_argument("--dataset", choices=list(KAGGLE_SOURCES), default="nrdd2024")
    ap.add_argument("--all", action="store_true", help="both datasets")
    ap.add_argument("--inspect-only", action="store_true",
                    help="download and report, stage nothing")
    ap.add_argument("--stage", metavar="DIR", help="stage into this directory")
    ap.add_argument("--s3", metavar="URI", help="also upload the staged result")
    ap.add_argument("--force", action="store_true",
                    help="proceed despite a schema or disk warning")
    ap.add_argument("--source-subdir", metavar="PATH",
                    help="stage from this directory instead of the auto-detected one")
    args = ap.parse_args()

    names = list(KAGGLE_SOURCES) if args.all else [args.dataset]
    rc = 0
    for n in names:
        # With --all, one --s3 prefix would overwrite the other dataset.
        s3 = args.s3
        if args.all and s3:
            s3 = s3.rstrip("/") + f"/{n}"
        rc |= process(n, args.stage, s3, args.inspect_only, args.force,
                      args.source_subdir)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
