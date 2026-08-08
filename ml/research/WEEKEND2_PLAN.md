# RDDS Detector — Weekend 2 Plan

**Window:** ~60 hours, SageMaker Unified Studio, `ml.g6.xlarge` (NVIDIA L4, 22 GB usable).
**Status of weekend 1:** E0 complete (3 seeds), E1 complete and **refuting**, E8 not run.
**What this weekend has to produce:** a second research leg to replace the cancelled E4,
and a measurably better model, both correctly compared.

Every hour figure below is derived from weekend 1's own `metrics.csv` files, not
extrapolated. The estimator was re-fitted against them and now predicts a 20-epoch
640 px run to within **1.8%** of the measured median.

---

## 0. Why the plan changed

E1 refuted the shape hypothesis (rho = +0.188, p = 0.607) and cancelled E4, which was
the programme's intended novel contribution. What remained was a clean benchmark and a
null result. That is publishable but thin, and it does not improve the deployed model.

Two things replace it, and they are chosen because the *data already points at them*:

**The generalisation leg (E10).** N-RDD2024 is six country archives concatenated, and
RDDS is deployed in Cluj, which is in none of them. "Train on five countries, test on
the sixth" is a benchmark the dataset has never had and simultaneously the exact
engineering question the deployment poses. This is the new contribution.

**The data-side ablations (E9, E8).** Weekend 1's per-class table is the strongest
unexplained signal in the project:

| Class | test AP@50 | note |
|---|---:|---|
| manhole_cover | 0.811 | **not road damage** — the product discards it |
| alligator_crack | 0.577 | |
| longitudinal_crack | 0.490 | |
| lane_line_blur | 0.477 | marking, capped at S1 by the pipeline |
| transverse_crack | 0.468 | |
| pedestrian_crossing_blur | 0.459 | marking, capped at S1 |
| pothole | 0.442 | |
| patchy_road | 0.354 | |
| repaired_crack | 0.340 | |
| rutting | **0.000** | 12 / 3 / 3 instances — unmeasurable |

The model's best class is the one the product throws away, and its worst is a class with
eighteen instances in the entire dataset. Before any architecture work, find out how
much of that spread is simply the class frequency distribution showing through. E9
ablates the oversampling constant nobody has ever measured; E8 asks whether the
discarded classes are consuming capacity or acting as useful hard negatives.

E4 stays cancelled. Do not revive it because the weekend feels short of novelty.

---

## 1. State recovery — do this first

### 1a. The research harness is not in git. Nothing in this plan runs until it is.

Commit `55e2f3e` removed `ml/aws/` and `ml/research/` from the repository. HEAD's `ml/`
tree now contains **14 files**, and `stage_dataset.py`, `weekend.py`, `experiments.py`,
`class_sets.py`, `compare.py`, `visualise.py` and the new `geo_splits.py` are **none of
them**. `train_experiment.py` and `repro.py` survived; the tooling around them did not.

This matters because §2 of RUNBOOK_20H.md starts the Studio session with `git clone`. That
clone currently produces a repo in which not one command in this plan exists. Weekend 1
worked because the files were present on the instance; a fresh clone will not be.

Fix before anything else — they are gitignored, so `-f` is required:

```bash
git add -f ml/aws/ ml/research/ ml/detection/train_experiment.py ml/repro.py
git commit -m "restore research harness to git so the Studio clone contains it"
git push
```

Then verify from a scratch directory, because this is the failure that costs a morning:

```bash
git clone <REPO-URL> /tmp/clonecheck && ls /tmp/clonecheck/ml/research/
```

`experiments.py`, `geo_splits.py`, `class_sets.py` and `compare.py` must all be listed.

### 1b. `runs/research/` was not on disk either

It was removed from HEAD on 6 August (`512614a`) and the working copy went with it. All 41
artefacts have been restored from `74edc9b`, which is a strict superset of every other
commit that touched the tree (`ca2f388`, `119fe57`, `fd28d89`, `b08399b`, `36b3404`,
`2395f75` — verified, nothing unique in any of them).

### 1c. A stale lock is blocking commits

```cmd
del .git\index.lock
```

Zero-byte leftover. Then back `runs/research/` up somewhere that is neither this disk nor
`.gitignore`d — weights retrain from the recorded seed and config, a lost `metrics.csv`
costs GPU hours that cannot be recovered.

---

## 2. What changed in the code

| File | Change | Why |
|---|---|---|
| `ml/research/geo_splits.py` | **New.** Country identification, `loco_split`, `matched_random_holdout`, 26 self-tests | E10 needs country-aware splitting; nothing in the repo did this |
| `ml/aws/stage_dataset.py` | dHash replaces aHash; grid configurable; banded Hamming search; **collapse guard**; `--calibrate-hash`; `--holdout-country` / `--holdout-control`; country distribution in the manifest | fixes the weekend-1 staging failure and adds the LOCO entry point |
| `ml/research/experiments.py` | L4-measured timing model; `data_variant` field + `DATA_VARIANTS`; E9 (3 specs) and E10 (6 specs); LOCO-needs-a-control invariant in `--check` | the estimator was wrong by ~2.6x on the phase that dominates a run |

Registry is clean at 31 experiments. Both new modules compile and self-test.

### The dHash finding — read this before staging

EXPERIMENTS.md §10 records dHash as the verified fix for the aHash collapse. **That is
not sufficient as stated, and acting on it directly would have cost hours again.**

Synthetic testing (clearly-labelled mock data: generated smooth low-contrast dashcam-style
frames, *not* N-RDD2024) reproduced the documented aHash failure exactly — 100% false-positive
pairs — and then found **dHash at grid 8 collapses too**, at ~90%. The cause is that a
bilinear downsample to 8×8 discards precisely the mid-frequency detail that distinguishes
two frames of the same road. Raising the grid helps but trades against robustness to benign
re-encoding, and no grid separated the two distance distributions cleanly on that fixture.

That fixture is harsher than the real archive, which spans six countries and many cameras.
So this is not a prediction that dHash will fail on N-RDD2024 — it is a reason not to
assume it will succeed. Two things now protect the run:

- **A collapse guard.** Staging aborts if the near-duplicate pass absorbs more than 25%
  of SHA-distinct images, prints the offending pairs, and tells you how to override. This
  is the check weekend 1 did not have; it turns a silent wrong number into a loud stop.
- **`--calibrate-hash`.** Measures, on the real archive, whether *any* threshold separates
  genuine re-encodes from different images. Verified to return `separated: true` on a
  high-entropy fixture and `separated: false` with an 86.8% false-positive rate on the
  collapsing one.

**Decision rule:** run the calibration; use near-duplicate detection only if it reports
`separated: true`. Otherwise stage with `--hash none` (exact SHA-256 dedupe only), which
is what weekend 1 effectively did, and keep near-duplicate risk as a stated limitation.
Every `DATA_VARIANTS` recipe defaults to `--hash none` for this reason.

Bonus: `--hash none` keeps the test split byte-comparable to weekend 1, so the new
numbers sit directly against 0.1991.

---

## 3. Two calibration gates, before the queue starts

### Gate 1 — batch size (~30 min GPU)

`resolve_batch` returns **16** for a 22 GB L4 at 640 px. EXPERIMENTS.md §11 already flags
this as using under a quarter of the card. The whole queue is priced twice below because
this one measurement moves the total by 15 hours.

```bash
python - <<'EOF'
import torch, time
from ultralytics import RTDETR
for b in (16, 32, 48, 64):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    m = RTDETR("rtdetr-l.pt")
    t = time.time()
    m.train(data=DATA, epochs=1, batch=b, imgsz=640, device=0, amp=True, val=False)
    print(b, f"{time.time()-t:.0f}s", f"{torch.cuda.max_memory_allocated()/2**30:.1f}GB")
EOF
```

**If throughput gain ≥ 1.25×:** adopt the new batch, and re-baseline with 3 seeds
(Phase B below) — the batch change makes the old 0.1991 not directly comparable, so the
noise floor must be re-measured at the new setting.

**If gain < 1.25×:** keep batch 16, and Phase B shrinks to **one** clean seed. Weekend 1's
three seeds remain the noise floor (0.0039); the single clean run only clears the ⚠dirty
flag by showing it reproduces within that floor. Saves 5 hours.

Do not assume the win. `throughput_gain` defaults to 1.0 in the estimator on purpose.

### Gate 2 — hash separation (~10 min CPU)

```bash
python ml/aws/stage_dataset.py --source $SRC --calibrate-hash 300
```

Exit 0 → `separated: true`, use `--hash dhash --hash-threshold <recommended>`.
Exit 2 → use `--hash none` everywhere. Both are acceptable outcomes; only guessing is not.

---

## 4. Prep (~3 h, almost all CPU)

```bash
# clean tree first: a dirty tree makes every run unreportable
git add -A && git commit -m "weekend 2: geo splits, dhash staging, E9/E10 registry"

python ml/aws/fetch_kaggle.py --dataset nrdd2024 --inspect-only   # must say class ORDER ok
python ml/aws/fetch_kaggle.py --dataset nrdd2024 --stage /tmp/src

python ml/aws/stage_dataset.py --source /tmp/src --calibrate-hash 300   # Gate 2

# country composition — confirm it matches the manifest before building folds
python ml/research/geo_splits.py --source /tmp/src
```

Expected, from `runs/research/_kaggle_nrdd2024.json`:
japan 7198 (37.9%) · usa 4804 (25.3%) · norway 2803 (14.8%) · china 1977 (10.4%) ·
india 1221 (6.4%) · czech 992 (5.2%) · **total 18,995**.

Then build the nine splits. Each spec emits its own command:

```bash
python -c "
import sys; sys.path.insert(0,'.')
from ml.research.experiments import REGISTRY
seen=set()
for s in REGISTRY.values():
    if s.stage in ('E9','E10') and s.data_variant not in seen:
        seen.add(s.data_variant)
        print(s.stage_command('/tmp/src', f'/tmp/staged_{s.data_variant}'))
"
```

Every staging run must print `clean: true`. A LOCO staging additionally prints its
country histogram and reminds you which control size to pair it with.

---

## 5. The queue

20 epochs (10 frozen + 10 full) at 640 px unless stated — matching weekend 1 exactly, so
everything is comparable to 0.1991 ± 0.0039. The winner is re-trained to 60 epochs at the
end, which is the only converged number the weekend produces.

| Ph | Run | h @ b16 | h @ 1.4× | Core |
|---|---|---:|---:|:--:|
| B | E0-baseline s1337 (clean tree) | 2.63 | 1.91 | ✓ |
| B | E0-baseline s1338 | 2.63 | 1.91 | ✓ |
| B | E0-baseline s1339 | 2.63 | 1.91 | drop 4th |
| C | E9-oversamplenone | 2.63 | 1.91 | ✓ |
| C | E9-oversample60 | 2.63 | 1.91 | ✓ |
| C | E8-structural7 | 2.63 | 1.91 | ✓ |
| C | E8-cracks_merged | 2.63 | 1.91 | drop 3rd |
| D | E10-loco-norway | 2.24 | 1.63 | ✓ |
| D | E10-control-norway | 2.24 | 1.63 | ✓ |
| D | E10-loco-india | 2.46 | 1.79 | ✓ |
| D | E10-control-india | 2.46 | 1.79 | ✓ |
| D | E10-loco-czech | 2.49 | 1.81 | drop 2nd |
| D | E10-control-czech | 2.49 | 1.81 | drop 2nd |
| E | E3-1024x576 | 3.74 | 2.70 | ✓ |
| E | E3-800sq | 4.04 | 2.92 | **drop 1st** |
| F | winner @ 60 ep, 1024×576 | 12.48 | 8.95 | ✓ |
| | **GPU total** | **53.1** | **38.4** | |
| | **core only** | **38.8** | **28.0** | |
| | + prep & analysis (~4.5 h) | 57.6 | 42.9 | vs 60 h |

**Run the core queue.** At batch 16 the full queue leaves 2.4 hours of margin across a
60-hour window, which is not margin. Core is 43.3 h all-in and leaves 17. Add droppables
back only once Gate 1 delivers a win or you are demonstrably ahead of schedule.

**E9-oversample30 is not in the queue** — it is the same staging as E0-baseline, so
Phase B's clean re-baseline *is* E9's control arm. Reuse it rather than paying 2.6 h twice.

### Why this drop order

`E3-800sq` goes first because `1024×576` dominates it: fewer pixels (589,824 vs 640,000),
so it is *cheaper*, and it preserves the native 16:9 dashcam aspect instead of spending
~44% of the input budget on letterbox padding. If resolution helps, 1024×576 shows it at
lower cost; 800² only adds a second point on the curve.

The czech fold goes next because norway already covers the European case and india covers
the upper bound. `E8-cracks_merged` next: informative, but E8-structural7 answers the
question the product actually asks. The third E0 seed goes last among droppables — but
note that dropping it means quoting the old floor rather than a new one, so drop it only
if Gate 1 said "keep batch 16" and the old floor still applies.

Never drop: two clean E0 seeds, the norway LOCO pair, the 60-epoch winner.

---

## 6. How each result gets judged

**E9 (oversampling).** Judge on per-class AP for the *rare* classes, never aggregate mAP —
resampling trades common-class accuracy for rare-class accuracy and the aggregate hides
the trade. `rutting` has 12 training instances and stays unmeasurable at every level;
report it as unmeasurable rather than quoting a number. If AP is flat across none / 0.30 /
0.60, the tail is a labelling-volume problem and the honest recommendation is to drop the
class from the product, not to train harder on it.

**E8 (class sets).** Shared classes only. `compare.py` already suppresses cross-class-set
aggregates and will refuse to print one. A *drop* in crack AP when the marking classes are
removed is the more interesting outcome: it would mean lane lines were working as hard
negatives, which argues for keeping them as an auxiliary task rather than a target.

**E10 (LOCO).** Only ever as a difference:

```
domain_shift_effect = AP(E10-control-<country>) − AP(E10-loco-<country>)
```

The fold alone conflates domain shift with the smaller training set. `experiments.py
--check` now fails if a LOCO spec has no matching control, so this cannot be forgotten.
Japan is deliberately excluded — at 37.9% of the data its train-size confound is larger
than the effect being measured. Per-country test splits also have different class
distributions; mark any class with under ~30 test instances as unmeasurable.

**E3 (resolution).** Judge on AP for the thin classes (longitudinal, transverse) against
the compact ones. If everything improves equally, the gain is general capacity, not the
sub-pixel-crack mechanism the hypothesis claims.

**E5 (recall) — free, do it on the winner.** Recall 0.438 against precision 0.615 is the
confirmed operating problem and it needs no training run: sweep per-class thresholds on
*val*, evaluate once on test. Report F2 and false-positives-per-kilometre. State the cost
ratio and justify it; picking β=2 because it flatters the number is not a result.

---

## 7. Analysis and export

```bash
python ml/research/compare.py --runs runs/research
python ml/research/compare.py --runs runs/research --baseline E0-baseline --challenger E8-structural7
python ml/research/compare.py --runs runs/research --baseline E10-control-norway --challenger E10-loco-norway
python ml/research/visualise.py --runs runs/research --out runs/research/_figures
```

Export after **every** run, not at the end — the account disappears with no recovery:

```bash
AWS_PROFILE=personal python ml/aws/weekend.py --export-now s3://<PERSONAL-BUCKET>/rdds-research
git add -f runs/research/ && git commit -m "results $(date +%H:%M)" && git push
```

The lesson from §1 is that git is the safety net that actually worked. Push small text
artefacts often.

---

## 8. Reportability checklist

Unchanged from RESEARCH_PROGRAM.md §6, plus two additions this weekend introduces:

- [ ] Seed recorded, git SHA captured, **dirty flag false** (this is the one weekend 1 failed)
- [ ] Number comes from the test split, and the test split was never used for selection
- [ ] Per-class AP alongside every aggregate
- [ ] Across-seed sd or CI on any comparison; nothing below the noise floor called a result
- [ ] Dataset manifest hash matches the one in `run.json`
- [ ] **`hash_algo` and `split_mode` recorded in the manifest** — two runs staged differently
      are not comparable, and this is now the easiest thing to get silently wrong
- [ ] **Every LOCO number reported with its control**, never alone

---

## 9. Known risks

- **The batch win may not materialise.** Dataloader-bound training will not speed up with
  a larger batch. Gate 1 exists to find out in 30 minutes rather than at hour 40.
- **Pixel-linear cost scaling is approximate.** Attention cost is not exactly linear in
  pixels; treat non-640 estimates as ±20% and re-measure after the first epoch at a new
  resolution.
- **20 epochs is still not convergence.** Comparisons remain valid; absolute numbers stay
  understated. Only the Phase F run is converged, and it is the only one to quote as an
  absolute figure.
- **LOCO test splits are small.** Czech is 992 images and India 1,221, so per-class AP on
  a rare class in those folds will be noisy. Report instance counts next to every AP.
- **`ml/research/`, `ml/aws/` and `runs/` are gitignored in full, and commit `55e2f3e`
  removed the harness from HEAD entirely.** §1a is the highest-priority item in this
  document: a fresh Studio clone currently contains none of the tooling this plan
  invokes. Force-add and verify with a scratch clone before booting the GPU instance.
