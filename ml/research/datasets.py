"""
ml/research/datasets.py
-----------------------
The dataset registry, and the class mapping between schemas.

Two datasets are wired up:

    nrdd2024   Kaya & Codur, 2024. doi:10.17632/27c8pwsd6v.3
               10 classes. The training set for everything in this programme.

    rdd2022    Arya et al., 2022. arXiv:2209.08538
               47,420 images from six countries. Public, and the benchmark the
               published literature (including ORDDC'2024) reports against.

WHY THE MAPPING NEEDS ITS OWN MODULE

    "Train on A, test on B" sounds like one line of config and is not. The two
    schemas are not the same, and the failure mode is silent: if the class ids do
    not line up, the evaluation still runs and still prints a number, it is just a
    number for the wrong classes. Every published cross-dataset road-damage result
    should be read with that in mind.

    Both datasets use the Japanese D-code convention, and the four codes they share
    have the SAME meaning:

        D00 longitudinal crack     D10 transverse crack
        D20 alligator crack        D40 pothole

    But the ids differ - D40 is index 4 in N-RDD2024 and index 3 in RDD2022's
    4-class arrangement - because N-RDD2024 inserts D30 (repaired_crack) before it.
    An off-by-one that maps "pothole" onto "alligator_crack" costs a fortnight if it
    is caught and a paper if it is not.

WHAT DOES NOT MAP, stated explicitly because it belongs in the paper

    N-RDD2024 has six classes with no RDD2022 equivalent in the challenge schema:
    repaired_crack, pedestrian_crossing_blur, lane_line_blur, manhole_cover,
    patchy_road, rutting. Cross-dataset transfer is therefore only measurable on
    the four shared classes, and any transfer claim has to say so. Use the
    `rdd2022compat` class set from ml/research/class_sets.py for both sides.

    Full RDD2022 also carries country-specific codes (D01, D11, D43, D44, ...) that
    the challenge subset excludes. Whether they appear depends on which country
    folders you download - check before assuming a 4-class layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = ["DatasetSpec", "DATASETS", "get_dataset", "shared_classes", "cross_map"]


@dataclass
class DatasetSpec:
    name: str
    citation: str
    doi_or_url: str
    classes: list[str]
    d_codes: dict[str, str]           # class name -> Japanese D-code
    n_images_approx: Optional[int] = None
    countries: str = ""
    notes: str = ""
    download_hint: str = ""
    default_s3_key: str = ""
    extra: dict = field(default_factory=dict)

    def index_of(self, class_name: str) -> Optional[int]:
        try:
            return self.classes.index(class_name)
        except ValueError:
            return None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "citation": self.citation, "doi_or_url": self.doi_or_url,
            "classes": self.classes, "n_classes": len(self.classes),
            "n_images_approx": self.n_images_approx, "countries": self.countries,
            "notes": self.notes,
        }


DATASETS: dict[str, DatasetSpec] = {}


def _reg(d: DatasetSpec) -> DatasetSpec:
    DATASETS[d.name] = d
    return d


_reg(DatasetSpec(
    name="nrdd2024",
    citation="Kaya, O. & Codur, M. Y., 2024. N-RDD2024: Road damage and defects",
    doi_or_url="https://doi.org/10.17632/27c8pwsd6v.3",
    classes=[
        "longitudinal_crack", "transverse_crack", "alligator_crack", "repaired_crack",
        "pothole", "pedestrian_crossing_blur", "lane_line_blur", "manhole_cover",
        "patchy_road", "rutting",
    ],
    d_codes={
        "longitudinal_crack": "D00", "transverse_crack": "D10",
        "alligator_crack": "D20", "repaired_crack": "D30", "pothole": "D40",
        "pedestrian_crossing_blur": "D50", "lane_line_blur": "D60",
        "manhole_cover": "D70", "patchy_road": "D80", "rutting": "D90",
    },
    countries="verify from the Mendeley record - do not assume",
    notes=(
        "The training set for this programme. Its geographic composition has NOT "
        "been verified in this repo; check the Mendeley record before making any "
        "domain-shift claim about Cluj, because that claim depends entirely on where "
        "these images were taken."
    ),
    download_hint=(
        "Download the archive from the Mendeley DOI (it requires accepting terms, so "
        "no script does it for you), extract, then run ml/aws/stage_dataset.py."
    ),
    default_s3_key="nrdd2024/v1",
))

_reg(DatasetSpec(
    name="rdd2022",
    citation="Arya et al., 2022. RDD2022: A multi-national image dataset",
    doi_or_url="https://arxiv.org/abs/2209.08538",
    classes=["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"],
    d_codes={
        "longitudinal_crack": "D00", "transverse_crack": "D10",
        "alligator_crack": "D20", "pothole": "D40",
    },
    n_images_approx=47_420,
    countries="Japan, India, Czech Republic, Norway, United States, China",
    notes=(
        "The challenge schema is 4 classes; note the pothole index is 3 here and 4 in "
        "N-RDD2024. Full country archives may also contain D01/D11/D43/D44 etc., "
        "which the challenge subset excludes - inspect what you actually downloaded. "
        "ORDDC'2024 reports against this dataset, which is the only context in which "
        "a comparison to that leaderboard is meaningful."
    ),
    download_hint=(
        "Public: https://github.com/sekilab/RoadDamageDetector . Annotations are "
        "PASCAL VOC XML; ml/detection/data_prep/prep_rdd2022.py already exists in "
        "this repo for the conversion."
    ),
    default_s3_key="rdd2022/v1",
))


def get_dataset(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset '{name}'. Known: {', '.join(sorted(DATASETS))}"
        ) from None


def shared_classes(a: str, b: str) -> list[str]:
    """
    Classes two datasets genuinely share, matched by D-CODE rather than by name.

    Matching on the D-code is deliberate. Class names get translated, abbreviated
    and pluralised differently between releases; the D-code is the stable identifier
    both schemas inherit from the original Japanese convention.
    """
    da, db = get_dataset(a), get_dataset(b)
    codes_b = {code: name for name, code in db.d_codes.items()}
    out = []
    for name, code in da.d_codes.items():
        if code in codes_b:
            out.append(name)
    # Preserve the first dataset's ordering for reproducibility.
    return [c for c in da.classes if c in out]


def cross_map(src: str, dst: str) -> dict[int, int]:
    """
    Map class ids from `src`'s schema into `dst`'s, for the shared classes only.

    Use when evaluating a model trained on one dataset against the other's labels.
    Classes absent from the target are omitted, so the caller must restrict the
    evaluation to the shared set rather than pretending the missing ones scored zero.
    """
    ds, dd = get_dataset(src), get_dataset(dst)
    codes_dst = {code: name for name, code in dd.d_codes.items()}
    mapping: dict[int, int] = {}
    for i, name in enumerate(ds.classes):
        code = ds.d_codes.get(name)
        if code and code in codes_dst:
            j = dd.index_of(codes_dst[code])
            if j is not None:
                mapping[i] = j
    return mapping


def describe_transfer(src: str, dst: str) -> dict:
    """A full, printable account of what a src -> dst transfer can and cannot measure."""
    ds, dd = get_dataset(src), get_dataset(dst)
    shared = shared_classes(src, dst)
    return {
        "source": src,
        "target": dst,
        "shared_classes": shared,
        "n_shared": len(shared),
        "id_map_src_to_dst": cross_map(src, dst),
        "source_only": [c for c in ds.classes if c not in shared],
        "target_only": [c for c in dd.classes if c not in shared],
        "recommended_class_set": "rdd2022compat",
        "warning": (
            f"Only {len(shared)} of {len(ds.classes)} source classes exist in "
            f"{dst}. Any transfer number describes those {len(shared)} classes "
            f"only, and must be reported that way. Train BOTH sides with the "
            f"'rdd2022compat' class set so the comparison is like for like."
        ),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Dataset registry and schema mapping")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", metavar="NAME")
    ap.add_argument("--transfer", nargs=2, metavar=("SRC", "DST"))
    args = ap.parse_args()

    if args.show:
        d = get_dataset(args.show)
        print(json.dumps(d.to_dict(), indent=2))
        print(f"\ndownload: {d.download_hint}")
    elif args.transfer:
        print(json.dumps(describe_transfer(*args.transfer), indent=2))
    else:
        for d in DATASETS.values():
            n = f"{d.n_images_approx:,}" if d.n_images_approx else "?"
            print(f"{d.name:12s} {len(d.classes):2d} classes  ~{n:>7s} images  {d.countries}")
        print(f"\nshared nrdd2024 <-> rdd2022: {shared_classes('nrdd2024', 'rdd2022')}")
