# RDDS Detector — Experiment Log

**Status: weekend 1 complete.** Everything under a "What we expect" heading is a
prediction written *before* seeing the numbers, kept exactly as written — including the
wrong ones.

### Headline results

| | |
|---|---|
| **Baseline** | Test mAP50-95 **0.1991 ± 0.0039** (3 seeds, held-out test split, zero leakage) |
| **Seed-noise floor** | **0.0039** — differences below this are not results |
| **E1 hypothesis** | **REFUTED.** ρ = +0.188, p = 0.607. Shape does not predict per-class accuracy |
| **E4 contribution** | **CANCELLED** — its motivation did not survive E1 |
| **Operating problem** | Recall 0.438 vs precision 0.615. Confirmed on clean data. E5 is the strongest lead |
| **E8 class ablation** | Not run — ran out of window. Ready for weekend 2 |

The programme's intended novel contribution was killed by its own gate, before any GPU
time went into building it. That is the gate working, and it is the most useful thing
that happened this weekend.

Started 3 August 2026 · **Europe AI Summer Research Program**, in partnership with AWS
Run on **Amazon SageMaker Unified Studio**, `ml.g6.xlarge` (NVIDIA L4, 24 GB) — see §2

> **Paper:** these results are written up in `paper/RDDS_Benchmark_2026.tex`
> ("What a Held-Out Split Costs You"). This file is the working log; the paper is the
> narrative version. Where they disagree, the run artefacts under `runs/research/`
> are authoritative.

---

## 1. Why these experiments exist

The detector was already trained and deployed. What it did not have was a way to tell
whether any of its numbers were trustworthy. Seven things had never been done:

| Never done before | Why it matters |
|---|---|
| **A held-out test split** | Every number came from the validation split that 57 epochs of checkpoint selection ran against. It is optimistically biased by an unknown amount. |
| **Per-class AP for the 10-class model** | Only the aggregate existed. You cannot tell which classes work without it. |
| **A leakage check** | Nothing recorded whether the same image appeared in both train and validation. If it did, the score is inflated. |
| **Seed variance** | With one run per configuration, there was no way to know whether a difference between two models was real or luck. |
| **Class-subset ablation** | The model spends capacity on classes the product then discards. Nobody had measured the cost. |
| **Reproducibility metadata** | No seed, no git SHA, no dataset fingerprint. A number could not be traced to the code and data that produced it. |
| **Statistical testing** | Comparisons were "0.31 beats 0.29", with no notion of whether that gap exceeds noise. |

The old training script (`ml/detection/train.py`) is kept untouched for provenance.
All new work goes through `ml/detection/train_experiment.py`.

---

## 2. Compute and infrastructure — AWS

Everything in this document runs on **Amazon Web Services**. The work is part of the
**Europe AI Summer Research Program**, run in partnership with AWS, and the compute is
provided through that partnership.

### What is actually being used

| Service | Role here |
|---|---|
| **Amazon SageMaker Unified Studio** | The development environment. Project `admin-project-788757560353`. |
| **SageMaker Code Editor space** | VS Code-based IDE where the code is edited and the experiment queue is launched. |
| **`ml.g6.xlarge` compute** | NVIDIA L4, 24 GB VRAM, 100 GB EBS. **Every number in this document was produced on this instance.** |
| **Amazon S3** | Dataset staging and result export. |
| **Amazon EC2** | Indirect — the Studio space runs on EC2, but direct `ec2:RunInstances` is blocked by IAM policy on the event account. |
| **Kiro** | AWS's spec-driven agentic IDE. Project context lives in `.kiro/steering/`. |

### Event constraints, and how they shaped the engineering

These are not incidental — several design decisions in `ml/aws/` exist only because of
them.

- **GPU access comes from a temporary AWS account** provisioned for a fixed window and
  funded by the event, separate from any personal credits.
- **That account is destroyed when the window closes, with no recovery.** This is why
  `ml/aws/weekend.py` exports results after *every single run* rather than once at the
  end: an unexpected cutoff then costs one experiment instead of the whole weekend.
- **`ec2:RunInstances` is blocked.** Compute must be obtained through SageMaker Unified
  Studio, which is why the training queue runs on the attached GPU rather than
  submitting managed jobs.
- **A hard wall-clock deadline.** `weekend.py` schedules against it and refuses to start
  a run that cannot finish in the time left.
- **Personal accounts use the AWS Free Tier credit model** ($100, up to $200 with the
  onboarding activities). SageMaker draws directly on those credits — there is no
  separate free allowance for ML services — so the research weekend is where GPU-heavy
  work belongs.

### Built for AWS, ready but not exercised this weekend

| Component | What it does | Why it is idle |
|---|---|---|
| `ml/aws/launch.py` | Submits experiments as managed **SageMaker Training Jobs**, with managed spot instances and `/opt/ml/checkpoints` sync so an interruption costs one checkpoint interval | The event account blocks the classic training-job path; `weekend.py` runs on the attached GPU instead. This is the route for a personal account. |
| `ml/aws/stage_dataset.py --s3` | Uploads the staged dataset to **S3** so training jobs can pull it as an input channel | Data is local this weekend, and no jobs are being submitted |
| `ml/tracking.py` | Logs runs to the **managed MLflow** tracking server available in Studio | Deferred: the temporary account's MLflow and the personal-account export profile create a credential conflict not worth debugging against a deadline. Local CSV and `run.json` remain authoritative either way. |

### Cost

GPU time during the research weekend is **event-funded and consumes no personal
credits**. For reference, the full experiment programme on a personal account is
estimated at **~127 GPU-hours**, roughly $190 at on-demand `ml.g5.2xlarge` pricing and
substantially less on managed spot. Re-derive it any time with:

```bash
python ml/research/experiments.py --budget
```

Those estimates are computed from this project's own measured epoch times rather than
assumed — see `estimate_hours()` in `ml/research/experiments.py`.

---

## 3. The starting point

What the previous model achieved, taken from `runs/detect/nrdd_2024/results.csv`:

| Metric | Value |
|---|---|
| Validation mAP50 | 0.5637 |
| Validation mAP50-95 | 0.2945 |
| Precision | 0.6554 |
| Recall | **0.5337** |
| Epochs | 57 (converged) |
| Time per epoch | 740 s on an RTX 2050 (4 GB) |

Two problems are visible without running anything.

**Recall is below precision.** For a municipal survey that is the wrong way round: a
missed pothole costs more than a false alarm an operator dismisses with one click. The
project's own `ml/detection/evaluate.py` already noted this.

**These are validation numbers.** They were selected on the same split they are measured
on. The true held-out performance is unknown and is almost certainly lower.

---

## 4. The data we are actually training on

Source: **N-RDD2024** (Kaya & Çodur, doi:10.17632/27c8pwsd6v.3), Kaggle mirror
`sannyshankaranml/n-rdd2024`.

The archive contains 62,010 image files, but only **18,995 are unique annotated
images** — the same photographs ship three times over, once per annotation format
(COCO, VOC, YOLO). Only the YOLO copies have usable labels. There is also an official
test set with **no labels at all** (the challenge withholds them), which is why we carve
our own test split rather than using theirs.

It is a **six-country** dataset:

| Country | Images | Share |
|---|---:|---:|
| Japan | 7,198 | 37.9% |
| USA | 4,804 | 25.3% |
| Norway | 2,803 | 14.8% |
| China | 1,977 | 10.4% |
| India | 1,221 | 6.4% |
| Czech Republic | 992 | 5.2% |

### Our split

Built by `ml/aws/stage_dataset.py`, seed 1337, stratified by each image's rarest class:

| Split | Images |
|---|---:|
| train | 13,297 |
| val | 2,849 |
| test | 2,849 |

**Leakage: zero.** No image appears in more than one split, verified independently of
the code that built the split.

### Class distribution (instances)

| Class | train | val | test |
|---|---:|---:|---:|
| longitudinal_crack | 16,104 | 3,335 | 3,356 |
| transverse_crack | 7,049 | 1,563 | 1,495 |
| alligator_crack | 5,473 | 1,191 | 1,156 |
| manhole_cover | 2,704 | 617 | 555 |
| lane_line_blur | 2,289 | 472 | 482 |
| pedestrian_crossing_blur | 1,937 | 403 | 399 |
| pothole | 1,576 | 343 | 353 |
| repaired_crack | 1,502 | 316 | 292 |
| patchy_road | 384 | 97 | 85 |
| **rutting** | **12** | **3** | **3** |

**`rutting` is unmeasurable.** Twelve training instances cannot teach a class, and AP
over three test instances is quantised to roughly {0, 0.33, 0.67, 1.0}. Whatever number
appears for rutting means nothing and will be reported as such. This is a property of
the dataset, not of our split — and it is independent evidence that a 10-class schema
does not match this data.

---

## 5. Training configuration

Identical for every experiment, so that any difference between runs is caused by the one
thing that was varied.

| Setting | Value |
|---|---|
| Model | RT-DETR-L (Zhao et al., arXiv:2304.08069) |
| Starting weights | COCO-pretrained |
| Input size | 640 × 640 |
| Batch | 16 |
| Epochs | 20 (10 backbone-frozen, then 10 full fine-tune at 0.1× LR) |
| Optimiser | AdamW, cosine LR |
| Hyperparameters | PSO-optimised values from `ml/optimization/pso_best.json` |
| Precision | fp16 (AMP) |
| Seeds | 1337, 1338, 1339 |

Measured on the L4: **832 iterations/epoch, ~2.6 it/s, ~5.5 min/epoch**, so roughly
**1 h 50 m per run**.

### Two honest caveats about this configuration

**20 epochs is not convergence.** The old model converged around epoch 50. We chose 20
so that four runs fit in a 19-hour window. Absolute scores will therefore sit below the
old 0.2945 — but the old number was measured on a different, possibly leaked split, so
it was never directly comparable anyway. What matters is that every run gets the same
budget, which keeps comparisons between them valid.

**Mosaic augmentation is effectively disabled.** Ultralytics turns mosaic off for the
final 10 epochs (`close_mosaic=10`), and each of our phases is exactly 10 epochs. The
PSO-tuned `mosaic=0.86` therefore does nothing. This applies identically to all runs, so
comparisons hold, but it should be fixed next time by setting `close_mosaic=0`.

---

## 6. The experiments

### E0 — Establish a real baseline and measure the noise floor

> ## RESULT — FINAL, all 3 seeds — 3 August 2026
>
> **Test mAP50-95 = 0.1991 ± 0.0039** (seeds 1337, 1338, 1339)
>
> | Seed | Test mAP50-95 |
> |---|---:|
> | 1337 | 0.19553 |
> | 1338 | 0.19861 |
> | 1339 | 0.20321 |
> | **mean ± sd** | **0.1991 ± 0.0039** |
>
> **Seed-noise floor: 0.0039.** Any later difference below this is not a result.
> Bootstrap 95% CI for the mean: [0.1955, 0.2032].
>
> | Metric | Test | Val |
> |---|---:|---:|
> | mAP50 | 0.4417 | 0.4266 |
> | mAP50-95 | **0.1986** | 0.2007 |
> | Precision | 0.6155 | 0.5847 |
> | Recall | **0.4382** | 0.4339 |
> | F1 | 0.5119 | 0.4981 |
>
> *(seed 1338; seed 1337 gave mAP50-95 0.1955)*
>
> ### Prediction scorecard
>
> | Prediction | Actual | Verdict |
> |---|---|---|
> | Test mAP50-95 in 0.20–0.27 | **0.1991** | Just below the range — near miss |
> | Seed spread 0.005–0.015 | **0.0039** | Slightly tighter than predicted |
> | Recall still below precision | **0.438 vs 0.615** | Correct |
>
> The two-seed floor read 0.0022; the third seed widened it to 0.0039. A worked
> illustration of why two seeds are not enough to estimate a spread — and why the
> programme spent three runs on this rather than one.
>
> ### Recall remains the operating problem
>
> Precision 0.615, recall 0.438 — the same inversion the old model had, now confirmed
> on a clean held-out split. The detector finds well under half the damage present.
> **E5 is fully justified.**
>
> ### Per-class AP@50 is remarkably stable across seeds
>
> | Class | s1337 | s1338 |
> |---|---:|---:|
> | manhole_cover | 0.798 | 0.811 |
> | alligator_crack | 0.566 | 0.577 |
> | longitudinal_crack | 0.478 | 0.490 |
> | lane_line_blur | 0.451 | 0.477 |
> | transverse_crack | 0.460 | 0.468 |
> | pedestrian_crossing_blur | 0.430 | 0.459 |
> | pothole | 0.417 | 0.442 |
> | patchy_road | 0.365 | 0.354 |
> | repaired_crack | 0.339 | 0.340 |
> | rutting | 0.000 | 0.000 |
>
> The **ordering is identical** across seeds and every value moves by less than 0.03.
> Per-class differences here are structural properties of the data, not sampling noise
> — which is what makes the `manhole_cover` gap (0.80 against 0.34–0.58 for every
> damage class) worth investigating rather than dismissing.
>
> ### Caveats
>
> - **Runs are flagged "dirty working tree"** by the harness: uncommitted edits were
>   present when they launched, so the recorded git SHA does not exactly describe the
>   code that ran. The code was functionally identical across all three, so
>   comparisons between them hold, but a paper should re-run from a clean commit.
> - **Not comparable to the old 0.2945.** That was 57 epochs on a validation split with
>   unverified leakage. This is 20 epochs on a clean held-out test split. Lower is
>   expected and does not indicate a worse model.
> - Third seed still running; the floor may shift slightly.

**Status: 2 of 3 seeds complete.** Original design follows.

**Question.** What does this model actually score on data it has never seen, and how much
does the score move between identical runs that differ only in random seed?

**Method.** Three runs, identical in every way except seed (1337, 1338, 1339). Evaluate
on the held-out test split exactly once, at the end.

**Why it comes first.** The spread across those three seeds is the noise floor. Any later
difference smaller than it is not a result. Without this number, no comparison in the
whole programme can be judged, so it is worth 3 of the 4 available runs.

**What we expect.**
- Test mAP50-95 **below 0.2945**, plausibly 0.20–0.27. Lower because it is a genuine
  held-out split and because 20 epochs is short.
- Seed spread of roughly 0.005–0.015 mAP50-95.
- Recall still below precision.

**What would surprise us.** A seed spread above 0.02 would mean single-run comparisons
are meaningless and every later experiment needs three seeds — which would not fit the
compute budget, and we would have to say so.

**Deliverable.** First per-class AP this project has ever had on a clean test split.

---

### E8 — Does dropping the non-damage classes help the ones that matter?

**Status: running (1 seed, `structural7`).**

**Question.** Three of the ten classes are not road damage: `manhole_cover` is
infrastructure, `lane_line_blur` and `pedestrian_crossing_blur` are road markings. The
system already discounts the markings — `pipeline/detector.py` caps both at severity S1
regardless of confidence. So the model pays full price to learn categories the product
throws away. Does removing them improve the seven that remain?

**Method.** Identical to E0 in every respect except the class set: `structural7` keeps
longitudinal, transverse, alligator, repaired crack, pothole, patchy road, rutting.

**How it will be judged.** On **per-class AP for the shared classes only**. Aggregate mAP
across different class sets is meaningless — removing the hardest classes raises the mean
without improving a single prediction. `ml/research/compare.py` detects the mismatch,
suppresses the aggregate, and refuses to report it.

**What we expect.** Honestly uncertain, and both outcomes are interesting:

- *If shared-class AP goes up*: the removed classes were consuming capacity. Practical
  recommendation follows immediately — ship a 7-class model.
- *If shared-class AP goes down*: the removed classes were acting as **useful hard
  negatives**. A lane line is a long, thin, bright object that looks a great deal like a
  crack; labelling it may be exactly what stops the model calling it one. That would
  argue for keeping them as an auxiliary task rather than a target, and it is the more
  interesting finding.

---

### E1 — Are elongated classes harder to detect? (analysis, no training)

> ## RESULT: HYPOTHESIS REFUTED — 3 August 2026
>
> **Spearman ρ = +0.188, permutation p = 0.607** (100,000 permutations, 10 classes,
> E0-baseline seed 1337, test split).
>
> The sign is *positive*, meaning more elongated classes scored marginally **better** —
> the opposite of the prediction — and the correlation is nowhere near significant.
> Box area does not explain it either: ρ = −0.248, p = 0.491.
>
> **Neither shape nor size predicts per-class accuracy on this dataset.**
>
> **E4 (shape-aware detection) is cancelled.** Its entire motivation was this
> correlation. Building an orientation-aware decoder prior would now be solving a
> problem the data says does not exist.
>
> ### Why the prior was wrong
>
> The pre-registered guesses about which classes are geometrically elongated were
> simply incorrect once measured:
>
> | Class | Predicted | Measured median AR | Anisotropy | AP@50 |
> |---|---|---:|---:|---:|
> | manhole_cover | compact | **2.82** | 1.49 | **0.798** |
> | transverse_crack | elongated | 5.15 | 2.37 | 0.460 |
> | lane_line_blur | elongated | **1.04** | 0.42 | 0.451 |
> | longitudinal_crack | elongated | 0.58 | 0.85 | 0.478 |
> | pothole | compact | 1.57 | 0.68 | 0.417 |
> | alligator_crack | compact | 1.38 | 0.63 | 0.566 |
>
> Two assumptions collapsed. **`manhole_cover` is not circular in image space** — median
> aspect ratio 2.82, because a round cover viewed from a moving vehicle projects to an
> ellipse. It is the *most* anisotropic class after transverse cracks, and it is by far
> the best detected. **`lane_line_blur` is essentially square** (AR 1.04); the annotated
> boxes cover marking patches, not long line segments. And a long crack's bounding box
> is only mildly elongated (AR 0.58) because real cracks meander rather than running
> straight.
>
> The lesson generalises: geometric intuition about a class should be **measured before
> a research programme is built on it**, not assumed from the class name. This cost a
> few hours of analysis; had E1 not been placed before E4, it would have cost days of
> GPU time and a wrong claim in the write-up.
>
> ### The question that replaces it
>
> `manhole_cover` reaches **0.798** while every damage class sits between 0.34 and 0.57.
> It has a sixth of `longitudinal_crack`'s training data, and it is neither the largest
> nor the most compact class — 81% of its boxes are under 1% of image area.
>
> What separates it is that it is a **manufactured object with a consistent physical
> boundary**, while road damage is amorphous and its extent is a judgement call by the
> annotator. The two worst classes, `repaired_crack` (0.339) and `patchy_road` (0.365),
> are the two most subjective categories in the schema.
>
> This suggests the ceiling is **annotation consistency**, not architecture. It is a
> hypothesis, not a result — it has not been tested, and no statistic here supports it
> yet.
>
> **Programme response, per this section's own gate:** pivot to E3 (resolution) and E5
> (operating point) as the main line.
>
> Artefacts: `runs/research/E1_anisotropy/`

**Status: COMPLETE. Result above.** Original design follows.

**Question.** Road damage classes are defined by *shape and orientation*, not appearance:
longitudinal cracks run along the road, transverse across it, rutting is long and
parallel, while potholes and alligator cracking are compact. A detector built on square
kernels and square input may be structurally mismatched to elongated targets.

**Method.** `ml/research/anisotropy.py`. For each class, compute the median
|log2(pixel aspect ratio)| of its boxes, then correlate that against per-class AP with a
Spearman rank test. The p-value comes from a 100,000-sample permutation test, which is
the right choice at only 10 classes.

**Why we suspect it.** The older 5-class RDD-2022 evaluation shows exactly the pattern
this predicts:

| Class | AP@50 | Shape |
|---|---:|---|
| pothole | 0.313 | compact |
| alligator_crack | 0.231 | compact |
| longitudinal_crack | **0.174** | elongated |
| transverse_crack | **0.126** | elongated |

The two compact classes are the two best; the two elongated ones are the two worst. That
is four points on a different dataset — suggestive, not proof, which is why it is being
tested properly.

**The sharpest version of the question.** `longitudinal_crack` has **16,104** training
instances; `pothole` has **1,576** — a 10× advantage. If longitudinal still scores lower,
data volume cannot explain it and shape becomes the leading candidate.

**Confound we will check.** Elongated boxes are also often small. `anisotropy.py`
correlates AP against box *area* as well, and reports whether shape and size are
themselves entangled. If they are, we say so rather than claiming shape.

**This gates everything after it.** A strong negative correlation justifies E4 (below). A
weak one means the elongation story is wrong and the programme pivots — which is also a
publishable finding.

---

### E3 — Does higher resolution recover thin cracks?

**Status: queued if time allows (~2 h).**

**Question.** At 640×640 a hairline crack is close to sub-pixel. Are thin cracks being
destroyed by the input pipeline before the model ever sees them?

**Method.** Same everything, input at 800×800.

**What we expect.** A gain, and a *larger* gain on the thin classes than the compact ones.
This is the least glamorous experiment and probably the best return per GPU-hour. If it
recovers most of the gap, then any later architectural claim has to be measured on top of
it rather than instead of it.

---

### E4 — Shape-aware detection *(next weekend, only if E1 supports it)*

**Status: not started. Gated on E1.**

**The intended contribution.** Three independent groups have found that strip-shaped
receptive fields help road damage detection — StripRFNet (arXiv:2510.16115), SPCNet, and
a length-aware cascade. All three did it inside a **CNN**.

Detection transformers have a mechanism CNNs do not: learned object queries. No published
work biases those queries with a shape prior. That is the gap, and it is
architecture-specific rather than a port of somebody else's module.

Three variants, ablated separately:

- **E4a** — strip convolutions (1×k, k×1) in the RT-DETR encoder. This is the *control*,
  not the contribution: it measures how much is explained by strip receptive fields alone.
- **E4b** — orientation-aware decoder query initialisation. **The novel part.**
- **E4c** — an aspect-ratio penalty in the box regression loss.

**How it will be judged.** Not on overall mAP. On per-class AP for
`{longitudinal_crack, transverse_crack}` specifically. If overall mAP rises but those
classes do not, the stated mechanism is wrong and we report that.

---

### E6 — Does it generalise to a country it has never seen?

**Status: designed, not started.**

Originally planned as a transfer to RDD2022. The country breakdown makes a better version
possible **inside this dataset**: train on five countries, test on the held-out sixth.
Stronger than a cross-dataset comparison, needs no extra download, and maps directly onto
the real deployment problem — Cluj is not in this dataset either.

---

### E5 — Fixing the recall/precision imbalance

**Status: designed, not started.**

Recall (0.534) sits below precision (0.655), which is backwards for municipal survey. Plan:
per-class thresholds tuned against F-beta with β=2, plus class re-weighting for the rare
classes. Report **false positives per kilometre** alongside F2 — a number a municipality
can actually act on. The cost ratio will be stated and justified rather than chosen
because it flatters the result.

---

## 7. Weekend 1 outcome — what was and was not achieved

The first research weekend ended 4 August 2026. Summary of the ledger:

| Experiment | Seeds | Outcome |
|---|---|---|
| **E0-baseline** | 1337, 1338, 1339 | **Complete.** Test mAP50-95 0.1991 ± 0.0039 |
| **E1-anisotropy** | analysis | **Complete.** Hypothesis refuted, ρ = +0.188, p = 0.607 |
| **E8-structural7** | — | **Not run.** Ran out of window |
| E3, E5, E6 | — | Not started |
| E4 | — | **Cancelled** by E1's result |

### Delivered

1. **A trustworthy baseline.** Three seeds, held-out test split, zero leakage, full
   per-class AP. The project had none of this before.
2. **A seed-noise floor of 0.0039**, so future comparisons are judgeable.
3. **A refuted hypothesis.** E1 killed the programme's intended contribution before a
   single GPU-hour was spent building it.
4. **A confirmed operating problem.** Recall 0.438 against precision 0.615 on clean
   data, which makes E5 the strongest remaining lead.
5. **Reproducible infrastructure** — staging with leakage proof, class-set derivation,
   deadline-aware scheduling, statistical comparison with guards.

### Not delivered, and why

**E8 (the class-selection question) did not run.** The queue's own deadline logic was
given a conservative `--hours-left` and skipped it rather than risk an unfinishable
run. Everything it needs is already fixed in place — the staged dataset, the split
manifest, and a 3-seed baseline at the identical budget — so it is a single ~3 h run
next weekend, directly comparable.

### Time lost, honestly accounted

Roughly 5 of ~19 hours went to problems rather than experiments:

| Cause | Cost |
|---|---|
| Ultralytics rewrote a relative `project` path, so phase 2 never ran | ~2.5 h (2 runs) |
| A killed parent left its training child on the GPU; contention halved throughput | ~1 h |
| aHash false-positive collapse forced a re-stage | ~1 h |
| GPU-hour estimates were extrapolated rather than measured | planning churn |

All four are fixed in the code. The first two are the kind of failure that only shows
up under a deadline, which is an argument for a dry run before the next weekend rather
than after it.

---

## 8. How the results are judged

Three rules, enforced in code rather than left to memory:

**A difference smaller than the seed noise is not a result.** `compare.py` measures the
noise floor from E0's three seeds and refuses to call anything below it a win.

**Aggregate mAP is never compared across different class sets.** When class counts
differ, `compare.py` suppresses the aggregate entirely and judges only the shared
classes. `visualise.py` mirrors this — bars are coloured by class count and carry an
explicit warning.

**Comparisons use a paired test over classes.** Per-class AP for two models is naturally
paired, so a paired permutation test over those pairs is used rather than comparing two
aggregate numbers. Where the sample is too small for significance (for example the 2–3
"thin" classes), the tool says so and reports the effect descriptively instead of quoting
a p-value it cannot support.

Commands:

```bash
python ml/research/compare.py --runs runs/research \
    --baseline E0-baseline --challenger E8-structural7
python ml/research/visualise.py --runs runs/research --out runs/research/_figures
```

---

## 9. Artefacts produced by each run

```
E0-baseline   seed 1337   10 classes   ~1h50m
E0-baseline   seed 1338   10 classes   ~1h50m
E0-baseline   seed 1339   10 classes   ~1h50m
E8-structural7 seed 1337   7 classes   ~1h50m
                                       ─────
                                       ~7.5 h
```

Each run produces, under `runs/research/<timestamp>_<experiment>_s<seed>/`:

| File | Contents |
|---|---|
| `run.json` | git SHA, seed, GPU, dataset fingerprint, timings |
| `config.json` | every resolved hyperparameter |
| `metrics.csv` | per-epoch train loss and validation metrics |
| `per_class_ap.json` | per-class AP@50 on the **test** split |
| `test_metrics.json` | the headline numbers |

---

## 10. Mistakes made and corrected

Recorded because they affect how the numbers should be read.

**Perceptual hashing collapsed on road images.** The first staging run reported 13,604
duplicates (72% of the dataset) and produced a 90/5/5 split. Testing showed aHash gives a
**100% false-positive rate** on low-contrast road imagery — an 8×8 greyscale average
carries almost no signal for uniform grey asphalt, and union-find then chains everything
into mega-groups. Near-duplicate detection is currently **disabled**; only exact
byte-identical duplicates are removed. dHash (gradient-based) was verified as the correct
replacement and is the fix for next time.

*Consequence:* if the dataset contains consecutive video frames of the same damage, some
may be split across train and test. Exact duplicates are excluded, so this is a
conservative gap rather than a known leak — but it is unverified, and it is stated rather
than assumed away.

**GPU-hour estimates were 5× wrong.** The plan assumed a 3.5× speedup over the original
RTX 2050. Measured on the L4, the real figure was ~1.4× at first, and only after fixing
the split did it reach the expected range. Every estimate is now derived from a measured
epoch time, not extrapolated.

**Class names were flagged as mismatched when they were not.** The Kaggle mirror labels
classes by Japanese D-code (D00…D90) rather than English names. The ordering was exactly
correct; the checker simply did not know D00 = longitudinal_crack. Fixed, and a genuinely
reordered class list is still blocked.

---

## 11. Known limitations

- **20 epochs is not convergence.** Absolute scores understate what this configuration can
  reach. Comparisons between runs remain valid.
- **`rutting` is unmeasurable** at 12 / 3 / 3 instances. Any AP reported for it is noise.
- **Mosaic augmentation is inactive** because each phase equals `close_mosaic`.
- **Near-duplicate detection is off** pending the dHash fix.
- **Only one seed for E8**, so its comparison against E0 rests on the noise floor E0
  measures rather than on its own repeats.
- **Batch size 16 uses under a quarter of the L4's memory.** Larger batches would train
  faster; the auto-sizing heuristic is too conservative for this model at 640px.

---

## 12. References

- RT-DETR — Zhao et al., 2023. [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- N-RDD2024 — Kaya & Çodur, 2024. [doi:10.17632/27c8pwsd6v.3](https://doi.org/10.17632/27c8pwsd6v.3)
- RDD2022 — Arya et al., 2022. [arXiv:2209.08538](https://arxiv.org/abs/2209.08538)
- StripRFNet — Lin et al., 2025. [arXiv:2510.16115](https://arxiv.org/abs/2510.16115)
- ORDDC'2024 — Arya, Omata et al., IEEE BigData Cup 2024
- COCO mAP — Lin et al., 2014. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
- PASCAL VOC AP — Everingham et al., 2010. doi:10.1007/s11263-009-0275-4
- Permutation tests — Ernst, 2004. doi:10.1214/088342304000000396

> **On ORDDC'2024's 86.18% F1:** that is *not* a comparable target. It is a 4-class
> RDD2022 problem whose winning method fine-tunes on pseudo-labels generated over the
> test set. Quoting it against a 10-class N-RDD2024 result would be a methodological
> error.

---

*Document written 3 August 2026, while the first experiments were running. It will be
updated with actual results — including any prediction above that turns out wrong.*
