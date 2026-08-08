"""
ml/research/class_sets.py
-------------------------
Train on a subset of the classes, or on merged classes, without duplicating the
dataset in S3.

WHY THIS IS A REAL EXPERIMENT AND NOT JUST PLUMBING

    Three of the ten N-RDD2024 classes are not road damage:

        manhole_cover              infrastructure annotation, not a defect
        lane_line_blur             a marking maintenance issue
        pedestrian_crossing_blur   a marking maintenance issue

    The system already treats the two marking classes as second-class citizens -
    pipeline/detector.py caps them at severity S1 regardless of confidence, because
    a faded crossing is not a structural problem. So the detector is spending
    capacity, and the classification loss is spending gradient, on categories the
    downstream product then discards.

    The question is whether removing them makes the seven classes a municipality
    actually acts on measurably better. That is a clean, cheap, defensible ablation,
    and it is the kind of finding a deployment paper can carry: "we removed three
    classes the product ignores and pothole AP rose by X".

    It can also go the other way. The marking classes may act as useful negatives -
    a lane line is a long thin bright thing that looks a lot like a crack, and
    labelling it may be what stops the model calling it one. If AP on the crack
    classes DROPS when markings are removed, that is an interesting result too, and
    it argues for keeping them as an auxiliary task.

HOW IT WORKS

    The dataset in S3 stays canonical: one upload, ten classes. The class set is
    applied inside the training container at job start, writing rewritten labels to
    local scratch and linking (not copying) the images. Deriving variants at run time
    rather than uploading one dataset per variant is the difference between a few
    seconds of setup and hours of transfer for every ablation.

THE EMPTY-IMAGE DECISION

    Dropping a class leaves some images with no boxes at all. Two options, and the
    choice changes what the experiment measures:

      keep (default)  the image stays as a pure background negative. The image count
                      is unchanged across class sets, so a comparison isolates the
                      effect of the CLASSES. The model still learns "there is nothing
                      here", which is what suppresses false positives.

      drop            the image is removed. Trains faster and the dataset is
                      "cleaner", but now two class sets differ in BOTH class count and
                      image count, and any difference between them is confounded.

    Default is `keep` because the confound is not worth the speed. Use `drop`
    deliberately, and say so when reporting.

Usage:
    from ml.research.class_sets import CLASS_SETS, materialise

    cs = CLASS_SETS["structural7"]
    yaml_path = materialise(src_root, dst_root, cs)

CLI:
    python ml/research/class_sets.py --list
    python ml/research/class_sets.py --apply structural7 \\
        --src /data/staged --dst /tmp/staged_structural7
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = ["ClassSet", "CLASS_SETS", "materialise", "NRDD2024_CLASSES"]

# The canonical N-RDD2024 ordering. Must match pipeline/detector.py CLASS_NAMES and
# the ids the weights were trained with - reordering this silently mislabels
# everything downstream.
NRDD2024_CLASSES: list[str] = [
    "longitudinal_crack",        # 0  D00  structural
    "transverse_crack",          # 1  D10  structural
    "alligator_crack",           # 2  D20  structural
    "repaired_crack",            # 3  D30  structural (repaired)
    "pothole",                   # 4  D40  structural
    "pedestrian_crossing_blur",  # 5  D50  marking
    "lane_line_blur",            # 6  D60  marking
    "manhole_cover",             # 7  D70  infrastructure
    "patchy_road",               # 8  D80  structural
    "rutting",                   # 9  D90  structural
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Definition
# ---------------------------------------------------------------------------
@dataclass
class ClassSet:
    """
    A view over the source classes: which to keep, which to merge, what to call the
    result.

    `keep` is an ordered list of OUTPUT class names. `merge` maps an output name to
    the source names that collapse into it. A name in `keep` with no `merge` entry
    passes through unchanged.
    """

    name: str
    description: str
    keep: list[str]
    merge: dict[str, list[str]] = field(default_factory=dict)
    drop_empty_images: bool = False
    rationale: str = ""

    # -- derived ------------------------------------------------------------
    def output_names(self) -> list[str]:
        return list(self.keep)

    def source_to_output(self, source_names: list[str]) -> dict[int, int]:
        """
        Map source class id -> output class id. Ids absent from the mapping are
        dropped.

        Output ids are contiguous 0..n-1 in `keep` order. This matters: YOLO and
        Ultralytics assume class ids index into `names` with no gaps, so keeping the
        original ids after removing a class in the middle would silently shift every
        later class by one. That is the single most common way a class-subset
        experiment produces garbage, so it is done once, here.
        """
        src_index = {n: i for i, n in enumerate(source_names)}
        mapping: dict[int, int] = {}

        for out_id, out_name in enumerate(self.keep):
            sources = self.merge.get(out_name, [out_name])
            for s in sources:
                if s not in src_index:
                    raise KeyError(
                        f"class set '{self.name}' references '{s}', which is not in "
                        f"the source schema {source_names}"
                    )
                mapping[src_index[s]] = out_id
        return mapping

    def dropped(self, source_names: list[str]) -> list[str]:
        kept_sources: set[str] = set()
        for out_name in self.keep:
            kept_sources.update(self.merge.get(out_name, [out_name]))
        return [n for n in source_names if n not in kept_sources]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "keep": self.keep,
            "merge": self.merge,
            "drop_empty_images": self.drop_empty_images,
            "n_classes": len(self.keep),
            "rationale": self.rationale,
        }


# ---------------------------------------------------------------------------
# The presets
# ---------------------------------------------------------------------------
CLASS_SETS: dict[str, ClassSet] = {}


def _reg(cs: ClassSet) -> ClassSet:
    CLASS_SETS[cs.name] = cs
    return cs


_reg(ClassSet(
    name="all10",
    description="All ten N-RDD2024 classes. The control.",
    keep=list(NRDD2024_CLASSES),
    rationale=(
        "The baseline every other class set is measured against. Identical to the "
        "current production configuration, so E0's numbers apply directly."
    ),
))

_reg(ClassSet(
    name="structural7",
    description="Structural road damage only: the three non-damage classes removed.",
    keep=[
        "longitudinal_crack", "transverse_crack", "alligator_crack",
        "repaired_crack", "pothole", "patchy_road", "rutting",
    ],
    rationale=(
        "Removes manhole_cover (infrastructure), lane_line_blur and "
        "pedestrian_crossing_blur (markings). These are the classes the product "
        "already discounts: pipeline/detector.py caps both marking classes at "
        "severity S1 regardless of confidence. If removing them raises AP on the "
        "seven that matter, the detector was spending capacity on categories the "
        "system throws away. If it LOWERS crack AP, the markings were acting as "
        "useful hard negatives - a lane line is a long thin bright object that "
        "resembles a crack - and that is an argument for keeping them as an "
        "auxiliary task rather than a target."
    ),
))

_reg(ClassSet(
    name="rdd2022compat",
    description="The four classes RDD2022 shares with N-RDD2024.",
    keep=["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"],
    rationale=(
        "D00/D10/D20/D40 are the classes the RDD2022 challenge evaluates. Training "
        "and evaluating on exactly this set is what makes an N-RDD2024 number "
        "comparable to published RDD2022 work, and it is the only honest basis for "
        "the E6 cross-dataset transfer experiment. Note this is also the class set "
        "the ORDDC'2024 leaderboard uses, so it is the one place a comparison to "
        "that literature is even meaningful."
    ),
))

_reg(ClassSet(
    name="cracks_merged",
    description="All crack subtypes collapsed into one 'crack' class.",
    keep=["crack", "pothole", "patchy_road", "rutting"],
    merge={
        "crack": [
            "longitudinal_crack", "transverse_crack",
            "alligator_crack", "repaired_crack",
        ],
    },
    rationale=(
        "Tests a different hypothesis from the subset ablations: that the model is "
        "not bad at cracks, it is bad at telling crack SUBTYPES apart. If merged "
        "crack AP is much higher than the mean of the four separate crack APs, the "
        "loss is in inter-subtype confusion rather than in detection, and the fix is "
        "a hierarchy (detect crack, then classify orientation) rather than a better "
        "detector. Worth knowing before building E4."
    ),
))

_reg(ClassSet(
    name="core4",
    description="The four highest-impact classes for municipal repair.",
    keep=["pothole", "alligator_crack", "rutting", "patchy_road"],
    rationale=(
        "The defects that actually generate work orders. A deployment-oriented "
        "class set: if this scores far better than all10, there is a case for "
        "running a small specialised model in the lite edge pipeline and the full "
        "model only in survey mode."
    ),
))


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------
def _link_or_copy(src: Path, dst: Path) -> str:
    """
    Make dst refer to src as cheaply as the filesystem allows.

    Symlink is free, hardlink is nearly free, copy is the fallback. For a dataset of
    tens of thousands of images this is the difference between seconds and minutes,
    repeated for every class-set variant.
    """
    if dst.exists() or dst.is_symlink():
        return "exists"
    try:
        os.symlink(src, dst)
        return "symlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    try:
        os.link(src, dst)
        return "hardlink"
    except (OSError, NotImplementedError, AttributeError):
        pass
    shutil.copy2(src, dst)
    return "copy"


def _rewrite_label(text: str, mapping: dict[int, int]) -> tuple[str, int, int]:
    """
    Rewrite one YOLO label file under the id mapping.

    Only the class id is touched. Box geometry is passed through byte-for-byte
    (as the original token strings), so no float round-trip can perturb a coordinate.
    Returns (new_text, n_kept, n_dropped).
    """
    out: list[str] = []
    kept = dropped = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
        except ValueError:
            continue
        if cid not in mapping:
            dropped += 1
            continue
        out.append(" ".join([str(mapping[cid]), *parts[1:]]))
        kept += 1
    return ("\n".join(out) + ("\n" if out else "")), kept, dropped


def materialise(
    src_root: Path | str,
    dst_root: Path | str,
    class_set: ClassSet,
    source_names: Optional[list[str]] = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    yaml_name: str = "dataset.yaml",
) -> Path:
    """
    Build a class-set view of a staged dataset at `dst_root`, returning the yaml path.

    Expects the layout stage_dataset.py produces:
        <src_root>/<split>/images/*.jpg
        <src_root>/<split>/labels/*.txt

    Images are linked, labels are rewritten. Nothing under src_root is modified.
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    source_names = source_names or NRDD2024_CLASSES
    mapping = class_set.source_to_output(source_names)

    if not src_root.is_dir():
        raise FileNotFoundError(f"source dataset not found: {src_root}")

    stats: dict[str, dict] = {}
    link_modes: dict[str, int] = {}

    for split in splits:
        s_img, s_lab = src_root / split / "images", src_root / split / "labels"
        if not s_img.is_dir():
            continue

        d_img, d_lab = dst_root / split / "images", dst_root / split / "labels"
        d_img.mkdir(parents=True, exist_ok=True)
        d_lab.mkdir(parents=True, exist_ok=True)

        n_img = n_empty = n_removed = 0
        boxes_kept = boxes_dropped = 0

        for img in sorted(s_img.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lab = s_lab / f"{img.stem}.txt"
            text = lab.read_text(encoding="utf-8", errors="replace") if lab.exists() else ""
            new_text, kept, dropped = _rewrite_label(text, mapping)
            boxes_kept += kept
            boxes_dropped += dropped

            if kept == 0:
                n_empty += 1
                if class_set.drop_empty_images:
                    n_removed += 1
                    continue

            mode = _link_or_copy(img, d_img / img.name)
            link_modes[mode] = link_modes.get(mode, 0) + 1
            # An empty label file is how YOLO encodes a pure background negative.
            (d_lab / f"{img.stem}.txt").write_text(new_text, encoding="utf-8")
            n_img += 1

        stats[split] = {
            "images_written": n_img,
            "images_with_no_boxes_after_filter": n_empty,
            "images_removed": n_removed,
            "boxes_kept": boxes_kept,
            "boxes_dropped": boxes_dropped,
        }

    if not stats:
        raise RuntimeError(
            f"no usable splits under {src_root} (looked for {splits}). Expected the "
            f"layout stage_dataset.py produces: <root>/<split>/images and /labels."
        )

    names = class_set.output_names()
    yaml_path = dst_root / yaml_name
    yaml_path.write_text(
        f"# Class set '{class_set.name}' derived from {src_root}\n"
        f"# {class_set.description}\n"
        f"# Dropped from source: {', '.join(class_set.dropped(source_names)) or '(none)'}\n"
        f"# Empty images after filtering: "
        f"{'REMOVED' if class_set.drop_empty_images else 'KEPT as background negatives'}\n"
        f"# Generated by ml/research/class_sets.py - do not hand-edit.\n"
        f"path: {dst_root}\n"
        + "".join(
            f"{s}: {s}/images\n" for s in splits if (dst_root / s / "images").is_dir()
        )
        + f"nc: {len(names)}\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)),
        encoding="utf-8",
    )

    (dst_root / "class_set.json").write_text(
        json.dumps({
            "class_set": class_set.to_dict(),
            "source_names": source_names,
            "source_to_output_id": {str(k): v for k, v in mapping.items()},
            "dropped_classes": class_set.dropped(source_names),
            "src_root": str(src_root),
            "per_split": stats,
            "link_modes": link_modes,
        }, indent=2),
        encoding="utf-8",
    )

    total_kept = sum(s["boxes_kept"] for s in stats.values())
    total_drop = sum(s["boxes_dropped"] for s in stats.values())
    print(f"[class_set] '{class_set.name}': {len(names)} classes, "
          f"{total_kept:,} boxes kept, {total_drop:,} dropped, "
          f"images linked via {link_modes or 'n/a'}")
    return yaml_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Class-subset views over a staged dataset")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", metavar="NAME")
    ap.add_argument("--apply", metavar="NAME")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--drop-empty", action="store_true",
                    help="remove images left with no boxes (confounds comparisons)")
    args = ap.parse_args()

    if args.show:
        cs = CLASS_SETS[args.show]
        print(json.dumps(cs.to_dict(), indent=2))
        print(f"\ndropped: {cs.dropped(NRDD2024_CLASSES)}")
        print(f"id map:  {cs.source_to_output(NRDD2024_CLASSES)}")
        return 0

    if args.apply:
        if not args.src or not args.dst:
            ap.error("--apply needs --src and --dst")
        cs = CLASS_SETS[args.apply]
        if args.drop_empty:
            cs = ClassSet(**{**cs.to_dict(), "drop_empty_images": True,
                             "keep": cs.keep, "merge": cs.merge})
        p = materialise(args.src, args.dst, cs)
        print(f"wrote {p}")
        return 0

    for cs in CLASS_SETS.values():
        dropped = cs.dropped(NRDD2024_CLASSES)
        print(f"{cs.name:16s} {len(cs.keep):2d} classes  {cs.description}")
        if dropped:
            print(f"{'':16s}    drops: {', '.join(dropped)}")
        if cs.merge:
            for out, srcs in cs.merge.items():
                print(f"{'':16s}    merges {' + '.join(srcs)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
