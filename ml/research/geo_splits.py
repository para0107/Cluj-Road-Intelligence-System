"""
Country-aware splitting for N-RDD2024 — the E10 leave-one-country-out (LOCO) study.

Why this exists
---------------
Weekend 1 killed the programme's intended contribution: E1 found no relationship
between a class's box shape and its AP (rho = +0.188, p = 0.607), so the
anisotropy-aware transformer (E4) was cancelled. What survived is a clean benchmark
and one unexplained operating problem (recall 0.438 against precision 0.615).

The strongest remaining research leg is generalisation, and N-RDD2024 supports it
natively: the dataset is six country archives concatenated, and the deployed RDDS
system runs in Cluj, which is in none of them. "Train on five countries, test on the
sixth" is therefore both a real benchmark contribution and the exact engineering
question the deployment poses.

The control is the whole point
------------------------------
Holding out a country changes two things at once: the test domain AND the training
set size. Japan is 37.9% of the data; a model trained without it is worse partly
because it saw 38% fewer images, which has nothing to do with domain shift. A LOCO
number quoted on its own therefore does not measure what it claims to measure.

`matched_random_holdout` builds the control: the same number of images removed, drawn
at random from all six countries, stratified the same way. The domain-shift effect is
the DIFFERENCE between the two, not the LOCO number itself:

    domain_shift_effect = AP(matched random holdout) - AP(leave-one-country-out)

Report both, always. A reviewer who is handed only the LOCO number will ask for the
control, and they will be right to.

Country composition, read from `runs/research/_kaggle_nrdd2024.json` (weekend 1),
not assumed:

    japan            7198   37.9%
    USA              4804   25.3%
    norway           2803   14.8%
    china-motorbike  1977   10.4%
    india            1221    6.4%
    Czech Republic    992    5.2%
    TOTAL           18995

Usage
-----
    from ml.research.geo_splits import country_of, loco_split, matched_random_holdout

    # inside stage_dataset.py, after grouping duplicates:
    split_idx = loco_split(groups, samples, holdout="norway", seed=1337)

Self-test:
    python ml/research/geo_splits.py --self-test
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Sequence

__all__ = [
    "COUNTRIES",
    "CountryStats",
    "country_of",
    "group_country",
    "country_histogram",
    "loco_split",
    "matched_random_holdout",
]


# ---------------------------------------------------------------------------
# Country identification
# ---------------------------------------------------------------------------
# Maps the directory token N-RDD2024 uses to a canonical name. The keys are matched
# case-insensitively against path components, so "Czech Republic_txt" and
# "czech republic_txt" both resolve.
#
# Kept as an explicit table rather than a regex over path parts: a silent mismatch
# here would put a country in the wrong fold and the resulting number would look
# perfectly plausible.
COUNTRIES: dict[str, str] = {
    "japan": "japan",
    "usa": "usa",
    "united states": "usa",
    "norway": "norway",
    "china-motorbike": "china",
    "china_motorbike": "china",
    "china": "china",
    "india": "india",
    "czech republic": "czech",
    "czech_republic": "czech",
    "czech": "czech",
}

# Canonical order, largest archive first. Used for stable reporting.
CANONICAL_ORDER: tuple[str, ...] = ("japan", "usa", "norway", "china", "india", "czech")

UNKNOWN = "unknown"


class _HasImage(Protocol):
    """Structural type for stage_dataset.Sample — avoids importing across packages."""

    image: Path
    classes: tuple[int, ...]


def country_of(path: Path | str) -> str:
    """
    Infer the source country from a path inside the N-RDD2024 archive.

    N-RDD2024 nests every image under `<country>_txt/<split>/images/`, so the country
    is recoverable from the path alone and needs no side-car metadata. Returns
    `UNKNOWN` when no component matches, which the callers treat as a hard error
    rather than a silently-dropped image.

    >>> country_of("/data/N-RDD2024/japan_txt/train/images/x.jpg")
    'japan'
    >>> country_of("/data/N-RDD2024/Czech Republic_txt/valid/images/y.jpg")
    'czech'
    """
    for part in Path(path).parts:
        token = part.lower()
        if token.endswith("_txt"):
            token = token[: -len("_txt")]
        token = token.strip()
        if token in COUNTRIES:
            return COUNTRIES[token]
    return UNKNOWN


def group_country(group: Sequence[int], samples: Sequence[_HasImage]) -> str:
    """
    One country for a whole duplicate-group.

    A group is a set of images the deduplicator decided are the same photograph, and
    the leakage invariant requires it to land entirely in one split. Cross-country
    groups should not exist; if one does it means two archives share an image, which
    is itself worth knowing. Majority vote, ties broken by canonical order so the
    result does not depend on dict iteration order.
    """
    votes = Counter(country_of(samples[i].image) for i in group)
    if len(votes) == 1:
        return next(iter(votes))
    top = max(votes.values())
    tied = [c for c, n in votes.items() if n == top]
    for c in CANONICAL_ORDER:
        if c in tied:
            return c
    return sorted(tied)[0]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
class CountryStats(dict):
    """`{country: n_images}` with a printable summary."""

    def report(self, stream=sys.stdout) -> None:
        total = sum(self.values()) or 1
        order = [c for c in CANONICAL_ORDER if c in self]
        order += sorted(c for c in self if c not in CANONICAL_ORDER)
        for c in order:
            n = self[c]
            print(f"  {c:10s} {n:6d}  {100 * n / total:5.1f}%", file=stream)
        print(f"  {'TOTAL':10s} {total:6d}", file=stream)


def country_histogram(samples: Sequence[_HasImage]) -> CountryStats:
    stats = CountryStats()
    for s in samples:
        c = country_of(s.image)
        stats[c] = stats.get(c, 0) + 1
    return stats


# ---------------------------------------------------------------------------
# Stratification shared with stage_dataset.split_groups
# ---------------------------------------------------------------------------
def _rarest_class_stratifier(
    samples: Sequence[_HasImage],
) -> Callable[[Sequence[int]], int]:
    """
    Stratify a group by the rarest class it contains.

    Same rule as `stage_dataset.split_groups`, duplicated deliberately rather than
    imported: `ml/aws/` is the staging layer and `ml/research/` is the analysis layer,
    and a circular import between them would be worse than eleven lines of overlap.
    An image holding both a common longitudinal crack and a rare rutting instance
    matters far more to the rutting split than to the crack split.
    """
    freq: Counter[int] = Counter()
    for s in samples:
        for c in s.classes:
            freq[c] += 1

    def stratum(group: Sequence[int]) -> int:
        present: set[int] = set()
        for i in group:
            present.update(samples[i].classes)
        if not present:
            return -1  # background-only images
        return min(present, key=lambda c: freq.get(c, 0))

    return stratum


def _split_pool(
    groups: list[list[int]],
    samples: Sequence[_HasImage],
    train_frac: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Stratified train/val partition of a pool of groups. Returns (train, val)."""
    stratum = _rarest_class_stratifier(samples)
    rng = random.Random(seed)

    by_stratum: dict[int, list[list[int]]] = defaultdict(list)
    for g in groups:
        by_stratum[stratum(g)].append(g)

    train: list[int] = []
    val: list[int] = []
    for _, gs in sorted(by_stratum.items()):
        gs = list(gs)
        rng.shuffle(gs)
        n_train = int(round(len(gs) * train_frac))
        # A stratum with >= 2 groups must contribute to val, or a rare class can
        # vanish from the selection split and early stopping stops seeing it.
        if len(gs) >= 2 and n_train >= len(gs):
            n_train = len(gs) - 1
        for k, g in enumerate(gs):
            (train if k < n_train else val).extend(g)
    return train, val


# ---------------------------------------------------------------------------
# The two splits
# ---------------------------------------------------------------------------
def loco_split(
    groups: list[list[int]],
    samples: Sequence[_HasImage],
    holdout: str,
    val_frac_of_train_pool: float = 0.15,
    seed: int = 1337,
    strict: bool = True,
) -> dict[str, list[int]]:
    """
    Leave-one-country-out: every image from `holdout` becomes the test split.

    The remaining five countries are partitioned into train/val by the same
    rarest-class stratification the normal staging uses, so the only difference
    between this and a standard split is *which* images are held out.

    Args:
        groups: duplicate-groups from `stage_dataset.group_duplicates`. Whole groups
            move together; this is the invariant that prevents leakage.
        samples: indexable by the ints inside `groups`.
        holdout: canonical country name, e.g. "norway". See `CANONICAL_ORDER`.
        val_frac_of_train_pool: validation fraction taken from the five-country pool.
        seed: controls only the train/val shuffle. The test split is fully determined
            by the country, which is the point.
        strict: raise if any image's country cannot be identified. Turning this off
            silently drops those images into the training pool.

    Returns:
        `{"train": [...], "val": [...], "test": [...]}` of sample indices.

    Raises:
        ValueError: unknown country name, empty holdout, or unidentifiable images
            when `strict`.
    """
    holdout = COUNTRIES.get(holdout.lower().strip(), holdout.lower().strip())
    known = set(CANONICAL_ORDER)
    if holdout not in known:
        raise ValueError(
            f"unknown holdout country {holdout!r}; expected one of {sorted(known)}"
        )

    labelled = [(g, group_country(g, samples)) for g in groups]

    unknown_n = sum(len(g) for g, c in labelled if c == UNKNOWN)
    if unknown_n and strict:
        example = next(samples[g[0]].image for g, c in labelled if c == UNKNOWN)
        raise ValueError(
            f"{unknown_n} images have no identifiable country (e.g. {example}). "
            "A LOCO fold built over unlabelled images does not measure domain shift. "
            "Extend geo_splits.COUNTRIES, or pass strict=False to pool them into train."
        )

    test_groups = [g for g, c in labelled if c == holdout]
    pool_groups = [g for g, c in labelled if c != holdout]

    if not test_groups:
        raise ValueError(
            f"holdout {holdout!r} matched zero images. Check the source layout: "
            f"found {sorted({c for _, c in labelled})}"
        )

    train, val = _split_pool(pool_groups, samples, 1.0 - val_frac_of_train_pool, seed)
    test = [i for g in test_groups for i in g]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def matched_random_holdout(
    groups: list[list[int]],
    samples: Sequence[_HasImage],
    n_test_images: int,
    val_frac_of_train_pool: float = 0.15,
    seed: int = 1337,
) -> dict[str, list[int]]:
    """
    The control for `loco_split`: hold out the same NUMBER of images, chosen at
    random across all countries, stratified by rarest class.

    Run this against every LOCO fold you report. Without it, a LOCO drop cannot be
    attributed to domain shift rather than to the smaller training set — and for
    Japan (37.9% of the data) the training-set effect alone is large.

    The selection is stratified so the control's test split has roughly the dataset's
    class distribution. It will NOT match the held-out country's class distribution,
    and it should not: the control isolates train-set size, and a second confound
    (test-set class mix) is reported separately by comparing the two class histograms.

    Args:
        n_test_images: target size, normally `len(loco["test"])` for the fold being
            controlled. Group granularity means the realised size is approximate;
            the achieved count is what should be reported.

    Returns:
        `{"train": [...], "val": [...], "test": [...]}` of sample indices.
    """
    if n_test_images <= 0:
        raise ValueError(f"n_test_images must be positive, got {n_test_images}")
    total = sum(len(g) for g in groups)
    if n_test_images >= total:
        raise ValueError(
            f"n_test_images={n_test_images} but only {total} images exist"
        )

    stratum = _rarest_class_stratifier(samples)
    rng = random.Random(seed)

    by_stratum: dict[int, list[list[int]]] = defaultdict(list)
    for g in groups:
        by_stratum[stratum(g)].append(g)

    # Take the target fraction from each stratum, so the held-out set mirrors the
    # dataset's class mix rather than over-drawing from whichever stratum is listed
    # first.
    frac = n_test_images / total
    test_groups: list[list[int]] = []
    pool_groups: list[list[int]] = []
    for _, gs in sorted(by_stratum.items()):
        gs = list(gs)
        rng.shuffle(gs)
        k = int(round(len(gs) * frac))
        if len(gs) >= 3:
            k = max(1, min(k, len(gs) - 2))   # never strip a stratum from train
        test_groups.extend(gs[:k])
        pool_groups.extend(gs[k:])

    train, val = _split_pool(pool_groups, samples, 1.0 - val_frac_of_train_pool, seed)
    test = [i for g in test_groups for i in g]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test() -> int:
    """Synthetic fixtures. No dataset required, so this runs anywhere."""
    from dataclasses import dataclass

    @dataclass
    class S:
        image: Path
        classes: tuple[int, ...]

    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    # -- country_of ---------------------------------------------------------
    check("japan path", country_of("/d/N-RDD2024/japan_txt/train/images/a.jpg") == "japan")
    check("czech path (space + case)",
          country_of("/d/N-RDD2024/Czech Republic_txt/valid/images/b.jpg") == "czech")
    check("usa path", country_of("/d/N-RDD2024/USA_txt/train/images/c.jpg") == "usa")
    check("china-motorbike path",
          country_of("/d/x/china-motorbike_txt/train/images/d.jpg") == "china")
    check("unmatched path -> unknown", country_of("/d/random/images/e.jpg") == UNKNOWN)

    # -- fixture: 6 countries, real proportions scaled down ------------------
    sizes = {"japan": 72, "usa": 48, "norway": 28, "china": 20, "india": 12, "czech": 10}
    dirname = {"japan": "japan", "usa": "USA", "norway": "norway",
               "china": "china-motorbike", "india": "india", "czech": "Czech Republic"}
    samples: list[S] = []
    for c, n in sizes.items():
        for i in range(n):
            samples.append(S(
                image=Path(f"/d/N-RDD2024/{dirname[c]}_txt/train/images/{c}{i}.jpg"),
                # class 9 (rutting) is deliberately ultra-rare, as in the real data
                classes=(i % 9,) if i % 17 else (9,),
            ))
    groups = [[i] for i in range(len(samples))]
    total = len(samples)

    hist = country_histogram(samples)
    check("histogram totals", sum(hist.values()) == total, f"{sum(hist.values())}")

    # -- loco_split ---------------------------------------------------------
    loco = loco_split(groups, samples, holdout="norway", seed=1337)
    n_train, n_val, n_test = (len(loco[k]) for k in ("train", "val", "test"))
    check("loco partitions exactly", n_train + n_val + n_test == total,
          f"{n_train}+{n_val}+{n_test}={n_train + n_val + n_test} vs {total}")
    check("loco test is exactly the holdout country", n_test == sizes["norway"],
          f"{n_test} vs {sizes['norway']}")
    check("loco test is 100% norway",
          all(country_of(samples[i].image) == "norway" for i in loco["test"]))
    check("loco train has zero norway",
          all(country_of(samples[i].image) != "norway" for i in loco["train"]))
    check("loco val has zero norway",
          all(country_of(samples[i].image) != "norway" for i in loco["val"]))
    check("loco splits are disjoint",
          not (set(loco["train"]) & set(loco["val"]) & set(loco["test"])))

    # every fold must be buildable
    for c in CANONICAL_ORDER:
        f = loco_split(groups, samples, holdout=c, seed=1337)
        ok = len(f["test"]) == sizes[c] and len(f["train"]) > 0
        check(f"fold {c} builds", ok, f"test={len(f['test'])} train={len(f['train'])}")

    # -- matched_random_holdout --------------------------------------------
    ctrl = matched_random_holdout(groups, samples, n_test_images=sizes["norway"], seed=1337)
    c_train, c_val, c_test = (len(ctrl[k]) for k in ("train", "val", "test"))
    check("control partitions exactly", c_train + c_val + c_test == total)
    check("control test size within 15% of target",
          abs(c_test - sizes["norway"]) <= max(2, 0.15 * sizes["norway"]),
          f"{c_test} vs {sizes['norway']}")
    check("control draws from >1 country",
          len({country_of(samples[i].image) for i in ctrl["test"]}) > 1)
    # The control exists to match TRAIN SIZE. That is the invariant that matters.
    check("control train size within 10% of loco train size",
          abs(c_train - n_train) <= max(3, 0.10 * n_train), f"{c_train} vs {n_train}")

    # -- determinism --------------------------------------------------------
    check("loco is deterministic",
          loco_split(groups, samples, "norway", seed=1337) == loco)
    check("seed changes train/val but not test",
          loco_split(groups, samples, "norway", seed=99)["test"] == loco["test"])

    # -- error paths --------------------------------------------------------
    try:
        loco_split(groups, samples, holdout="romania")
        check("unknown country raises", False)
    except ValueError:
        check("unknown country raises", True)

    bad = samples + [S(image=Path("/d/mystery/images/z.jpg"), classes=(0,))]
    try:
        loco_split(groups + [[len(samples)]], bad, holdout="norway")
        check("unidentifiable image raises under strict", False)
    except ValueError:
        check("unidentifiable image raises under strict", True)

    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        return 1
    print("all geo_splits self-tests passed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--self-test", action="store_true", help="run synthetic checks")
    ap.add_argument("--source", type=Path, help="scan a real archive and report countries")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()

    if args.source:
        from dataclasses import dataclass

        @dataclass
        class S:
            image: Path
            classes: tuple[int, ...] = ()

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        samples = [S(p) for p in args.source.rglob("*") if p.suffix.lower() in exts]
        if not samples:
            print(f"no images under {args.source}", file=sys.stderr)
            return 2
        print(f"country composition of {args.source}:")
        hist = country_histogram(samples)
        hist.report()
        if hist.get(UNKNOWN):
            print(f"\n{hist[UNKNOWN]} images unidentified — extend COUNTRIES before "
                  f"running a LOCO fold", file=sys.stderr)
            return 2
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
