# RDDS Detector — Research Program

**Target venue:** Europe AI Summer Research 2026 (project accepted; this document covers
the model-development phase that follows acceptance).
**Scope:** the detector only. Severity, depth, segmentation, and the platform are frozen
for the duration of this program.
**Compute:** Amazon SageMaker Unified Studio. Dataset staged to S3 (not held locally).

---

## 0. Ground truth about where we are

Everything in this table was read out of the repository, not assumed. Where a number
does not exist, that is stated rather than estimated.

| Item | Value | Source |
|---|---|---|
| Architecture | RT-DETR-L (Zhao et al., 2023) | `pipeline/detector.py` |
| Dataset | N-RDD2024, 10 classes | `runs/detect/nrdd_2024/dataset_nrdd2024.yaml` |
| Training length | 57 epochs, converged | `runs/detect/nrdd_2024/results.csv` |
| Final val mAP50 | **0.5637** | epoch 57 |
| Final val mAP50-95 | **0.2945** | epoch 57 |
| Final val precision | 0.6554 | epoch 57 |
| Final val recall | **0.5337** | epoch 57 |
| Per-class AP, 10-class model | **does not exist** | searched `runs/`, `ml/evaluation/` |
| Test-split result | **does not exist** | the yaml defines `train`/`val` only |
| Hyperparameters | PSO over 7 dims | `ml/optimization/pso_best.json` |
| PSO search history | **not in repo** | `pso_history.json` absent |
| Seeding / run manifests | **absent** | `train.py` has no `--seed`, `deterministic=False` |

Three facts follow from this and they set the agenda.

**Convergence is real, not premature.** mAP50 moved from 0.56348 to 0.56366 across epochs
55 to 57 with the learning rate down at 4.7e-6. Training longer on this configuration buys
nothing. The next gain has to come from data, resolution, or architecture — not schedule.

**Every number currently held is a validation number.** Fifty-seven epochs of checkpoint
selection ran against that same split. The reported 0.2945 is therefore optimistically
biased by an unknown amount. There is no held-out test set to quantify it. For a research
program this is the first thing that has to be fixed, because every later comparison
inherits the bias.

**Recall is below precision, and that is the wrong way round for this application.** The
project's own `ml/detection/evaluate.py` already says so in its header notes: a missed
pothole in a municipal survey costs more than a false alarm an operator dismisses in one
click. The model is currently tuned against its own deployment economics.

---

## 1. Where the field is, as of August 2026

| Work | Year | Relevance |
|---|---|---|
| RT-DETR (Zhao et al.) | 2023 | The current baseline. Two generations old. |
| RT-DETRv2 (Lv et al.), RT-DETRv4 | 2024, 2025 | Direct drop-in successors to the baseline. |
| ORDDC'2024 winner (Fujitsu) | 2024 | F1 **86.18%** @ 136.4 ms/img. Co-DETR + RTMDet teachers distilled into a YOLOv10 student, plus pseudo-labelling on the test set. |
| StripRFNet (Lin et al., arXiv 2510.16115) | Oct 2025 | YOLO11 base. Strip convolutions for slender cracks, LSKA shape-perception, P2 head for small objects. |
| SPCNet (strip pyramid ConvNeXt) | 2023 | Earlier strip-convolution treatment of road defects. |
| Length-aware cascade RDD | 2021 | Earliest explicit use of damage elongation as a prior. |

Two things to take from this.

The ORDDC number is **not** a target to chase. It is a 4-class RDD2022 problem, and the
winning method fine-tunes on pseudo-labels generated over the test set. That is a
competition technique, not a deployable one, and it is not comparable to a 10-class
N-RDD2024 result. Quoting it as a bar to clear would be a methodological error a reviewer
would catch immediately.

The strip-convolution line of work **is** the important signal. Three independent groups
have converged on the same observation: road damage classes are defined by elongation, and
standard square-kernel detectors are structurally mismatched to them. All three
implemented it inside a CNN. That leaves an opening, described in §3.

---

## 2. The three gaps worth attacking

### Gap A — N-RDD2024 has no rigorous public 10-class benchmark

A literature check found no published per-class results for the full 10-class N-RDD2024
schema. The dataset is on Mendeley with a DOI and is being used, but the community has no
agreed baseline to compare against.

A carefully constructed benchmark — held-out test split, several modern architectures under
one identical training protocol, per-class AP with confidence intervals, and a published
split manifest — is a citable contribution in its own right, and it is the least risky work
in this program. It also produces the measurement apparatus that everything else needs.

### Gap B — the model has no shape prior, and the data says it needs one

This is the substantive research contribution.

The N-RDD2024 classes are separated by *geometry and orientation*, not by appearance:

| Class | Geometry |
|---|---|
| D00 longitudinal crack | thin, elongated along the driving direction |
| D10 transverse crack | thin, elongated across the driving direction |
| D90 rutting | long, parallel, low contrast |
| D20 alligator crack | compact, texture-defined |
| D40 pothole | compact, blob-like |

The earlier 5-class RDD-2022 evaluation in `runs/evaluation/RDD-2022/` shows exactly the
failure pattern this predicts:

| Class | AP@50 | Geometry |
|---|---|---|
| pothole | 0.313 | compact |
| alligator_crack | 0.231 | compact |
| longitudinal_crack | **0.174** | elongated |
| transverse_crack | **0.126** | elongated |

The two elongated classes are the two worst. The two compact classes are the two best.

That is a suggestive correlation on four points from a different dataset, not proof — it is
a *hypothesis*, and E1 is designed to test it properly on N-RDD2024 before any architecture
work begins.

If it holds, the contribution is this: **strip-awareness has been demonstrated inside CNN
detectors (StripRFNet, SPCNet), but not inside a detection transformer.** DETR-family models
have a mechanism CNNs do not — learned object queries — and no published work initialises
or biases those queries with a shape prior. That is the novel angle, and it is a good one
because it is architecture-specific rather than a port of someone else's module.

### Gap C — nobody optimises for the asymmetric municipal cost

The RDD literature reports F1, which weights a missed crack and a false alarm equally. In a
municipal survey they are not equal. A recall-oriented treatment with an explicitly stated
and justified cost ratio, reported as an F-beta curve rather than a single F1, is a small
but honest contribution and it directly improves the deployed system.

---

## 3. The experiment ladder

Each experiment states a hypothesis, the measurement that would falsify it, and a gate that
decides whether to continue. Experiments run in order; the gates exist so that effort is not
spent on a clever idea when a boring one already explains the result.

### E0 — Reproduce and instrument

**Hypothesis:** the 0.2945 result is reproducible under seeding.
**Do:** re-run the current PSO configuration through the new harness with three seeds
(1337, 1338, 1339). Record per-class AP for the first time.
**Measure:** mean and standard deviation of mAP50-95 across seeds.
**Gate:** if seed variance exceeds roughly 0.01 mAP50-95, every subsequent single-run
comparison in this program is noise and all later experiments need three seeds, which
triples the budget. Find this out now, not in September.

### E1 — Dataset audit, honest splits, and the anisotropy evidence

**Hypothesis (the important one):** per-class AP is inversely related to the class's median
box aspect ratio. Elongated classes are detected worse.
**Do:**
1. Hash-based leakage check between `train_oversampled/`, `valid/`, and the new test split.
   The current training set is oversampled and there is no record of whether oversampling
   duplicated images across the split boundary. If it did, the 0.5637 is inflated and the
   whole baseline moves.
2. Carve a genuine held-out test split, stratified by class, and publish the manifest hash.
3. Per-class box statistics: aspect-ratio distribution, absolute size distribution, count.
4. Plot per-class AP (from E0) against median aspect ratio.

**Measure:** Spearman correlation between per-class AP and median |log aspect ratio|.
**Gate:** this is the decision point for the whole program. Strong negative correlation
means Gap B is real and E4 is justified. Weak or absent correlation means the elongation
story is wrong, and the program pivots to E2/E3/E5 as its main line. Either outcome is a
publishable finding, which is why this experiment is second and not seventh.

This step also produces the paper's motivating figure.

### E2 — Architecture bake-off

**Hypothesis:** part of the gap is simply that the baseline is two generations old.
**Do:** RT-DETR-L (baseline), RT-DETRv2, RF-DETR, YOLOv12, YOLO11 — identical data, schedule,
augmentation, seed. Only the architecture varies.
**Measure:** test mAP50-95, per-class AP, latency at the deployment resolution.
**Gate:** if a newer model gives more than about 2 points of mAP50-95 for free, adopt it as
the baseline *before* E4. There is no point demonstrating a novel module on top of an
outdated backbone; a reviewer will ask whether the gain survives on a modern one.

### E3 — Resolution and input aspect ratio

**Hypothesis:** thin cracks are being destroyed by the input pipeline before the model ever
sees them. At 640×640 a hairline crack is close to sub-pixel, and letterboxing 16:9 dashcam
frames into a square wastes roughly 44% of the input budget on padding.
**Do:** 640² (baseline), 800², 1024², and **non-square** 1024×576 preserving native dashcam
aspect. Also test the P2 high-resolution head that StripRFNet reports as significant.
**Measure:** test mAP50-95 overall and restricted to small objects; latency.
**Note:** this is the highest expected gain per unit of compute in the whole program, and it
is unglamorous. Run it before the interesting work, because if it recovers most of the gap
then E4's claim has to be measured on top of it, not instead of it.

### E4 — The contribution: anisotropy-aware detection transformer

Gated on E1. Three variants, ablated separately so the paper can attribute the gain.

- **E4a — strip receptive fields in the encoder.** Asymmetric 1×k and k×1 convolutions in
  the RT-DETR hybrid encoder. This is the StripRFNet idea ported into a transformer. On its
  own it is a replication, not a contribution, but it is the necessary control.
- **E4b — orientation-aware query initialisation.** *The novel component.* RT-DETR selects
  encoder features to initialise decoder queries. Bias that selection, and the query's
  positional prior, by an explicit anisotropy signal, so that a subset of queries is
  predisposed to elongated, oriented boxes. No published DETR-family work does this.
- **E4c — aspect-ratio-aware regression loss.** Augment the GIoU term with an explicit
  aspect-ratio penalty for the elongated classes, so a shape error on a thin crack costs
  more than the same IoU error on a pothole.

**Measure:** the honest one is **not** overall mAP. It is per-class AP on
`{longitudinal_crack, transverse_crack, rutting}` specifically, against the E2/E3 best.
**Gate:** if overall mAP rises but the elongated classes do not, the stated mechanism is
wrong and the gain came from somewhere else. Report that if it happens. A mechanism check
that fails and is reported honestly is worth more at a research program than a number that
went up for unexplained reasons.

### E5 — Recall-oriented operating point

**Hypothesis:** the model is better than its F1 suggests, and is simply thresholded for the
wrong cost function.
**Do:** per-class threshold selection against F-beta with β=2; class-balanced or focal
re-weighting for the rare classes identified in E1.
**Measure:** F2 and the full precision-recall operating curve, plus the recall achieved at a
fixed operator-review budget (false positives per kilometre — a number a municipality
actually cares about).
**Note:** state and justify the cost ratio explicitly. Do not silently pick β=2 because it
flatters the result.

### E6 — Cross-dataset generalisation

**Hypothesis:** a model trained on N-RDD2024 degrades on roads from a different country,
which is the situation the deployed system is actually in.
**Do:** train on N-RDD2024, test on the mappable RDD2022 subset, and the reverse. Map the
overlapping classes explicitly and document what does not map.
**Measure:** the drop, per class.
**Why it matters twice:** it is the question a reviewer will ask, and it is a real
engineering risk for the deployment, since the training data's geographic composition and
Cluj's road surfaces are not the same distribution. Verify the dataset's actual geographic
composition from the Mendeley record rather than assuming it.

### E7 — Deployment-constrained Pareto front

**Hypothesis:** the best research model is not the best deployable model.
**Do:** plot mAP50-95 against latency for every model in the program, at the edge target,
including the ONNX/fp16 paths the lite pipeline already uses.
**Why:** this connects the research to a system that is genuinely deployed with real users.
Almost no RDD paper can show that, and it is the strongest differentiator of this project
against a pure benchmark submission.

---

## 4. Risk portfolio

A research program should not be a single bet.

| Experiment | Expected gain | Risk | Role |
|---|---|---|---|
| E1 | none directly | very low | produces the evidence and the motivating figure |
| E3 | moderate to large | very low | the reliable win |
| E2 | moderate | very low | free check, prevents an embarrassing question |
| E5 | small, application-relevant | low | honest contribution, improves deployment |
| E6 | none (it is a measurement) | low | reviewer-proofing, real engineering value |
| **E4** | **potentially large** | **high** | **the actual contribution** |
| E7 | none (it is a measurement) | low | the differentiator |

If E4 fails, E1 + E2 + E3 + E5 + E6 + E7 still constitute a complete, publishable benchmark
and analysis paper on a dataset that has none. That is the point of the ordering: the
program cannot come away empty-handed.

---

## 5. Compute plan

Instance pricing changes; verify against the AWS pricing page before committing budget. As
of this writing SageMaker training on `ml.g5.2xlarge` (1×A10G, 24 GB) is listed at roughly
**$1.52/hour** on demand, with managed spot typically reducing that substantially. `ml.g6`
(L4) is the newer generation and should be price-checked directly.

Practical guidance for this program:

- **Single-GPU `ml.g5.2xlarge` is enough** for everything except E2's larger models. The
  baseline was trained on a 4 GB RTX 2050; 24 GB removes the batch-size and resolution
  constraints that shaped the original configuration. Several of the current hyperparameters
  exist only to fit 4 GB and should be revisited rather than inherited.
- **Use managed spot with checkpointing.** These are multi-hour, restartable jobs — the
  archetypal spot workload. The harness writes checkpoints to
  `/opt/ml/checkpoints` so an interruption costs one checkpoint interval, not one run.
- **Never let a sweep share a run directory.** Each trial gets its own, or the per-trial
  checkpoints are lost and the best configuration cannot be reconstructed.
- **Budget order:** E1 costs almost nothing (CPU-only analysis). E0 and E3 are cheap. E2 is
  the largest fixed cost. E4 is where the remaining budget should go.

---

## 6. What "done" means

A result is not reportable until all of the following hold. This is the checklist to run
before quoting any number in a paper or a talk.

- [ ] Seed is set and recorded in `run.json`
- [ ] Git SHA captured, and the dirty flag is false
- [ ] Full resolved config saved to `config.json`
- [ ] `metrics.csv` contains validation metrics per epoch, not only training loss
- [ ] The reported checkpoint is the one `best.pt` points at
- [ ] The number comes from the **test** split, and the test split was never used for
      selection
- [ ] Per-class AP reported alongside the aggregate
- [ ] A confidence interval or across-seed standard deviation accompanies any comparison
- [ ] The dataset manifest hash matches the one recorded in the run

---

## 7. Immediate next actions

1. **Stage the data and prove the splits are clean.**
   `python ml/aws/stage_dataset.py --source <nrdd> --out <staged> --dry-run`
   Do this before spending a single GPU-hour. If the dry run reports duplicates in the
   source, the original `train_oversampled/` vs `valid/` boundary is suspect and the
   0.5637 baseline moves. Then re-run without `--dry-run` and with `--s3`.

2. **Audit the data.** Run the `dataset-audit` skill for class balance, box sizes,
   corrupt files and cross-split leakage. It covers the general checks;
   `ml/research/anisotropy.py` covers the shape analysis specific to this program and
   does not duplicate them.

3. **Run E0** — `python ml/aws/launch.py --experiment E0-baseline --data s3://… --output s3://…`
   Three seeds. Produces the first per-class AP for the 10-class model, the first test-split
   number, and the seed-noise floor that every later comparison is judged against.

4. **Test the hypothesis.**
   `python ml/research/anisotropy.py --labels <staged>/test/labels --images <staged>/test/images --per-class-ap <run>/per_class_ap.json`
   Read the correlation, then decide whether E4 is on. Everything downstream branches here.

5. **Compare, always.** `python ml/research/compare.py --runs runs/research --baseline <a> --challenger <b>`
   Never quote a difference that this tool has not confirmed exceeds the seed-noise floor.

---

## References

- Zhao et al., 2023. *DETRs Beat YOLOs on Real-time Object Detection* (RT-DETR).
  [arXiv:2304.08069](https://arxiv.org/abs/2304.08069)
- Lv et al., 2024. *RT-DETRv2: Improved Baseline with Bag-of-Freebies*.
  [arXiv:2407.17140](https://arxiv.org/abs/2407.17140)
- *RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation
  Models*, 2025. [arXiv:2510.25257](https://arxiv.org/abs/2510.25257)
- Lin et al., 2025. *StripRFNet: A Strip Receptive Field and Shape-Aware Network for Road
  Damage Detection*. [arXiv:2510.16115](https://arxiv.org/abs/2510.16115)
- Arya, Omata et al., 2024. *ORDDC'2024: State of the art Solutions for Optimized Road
  Damage Detection*. IEEE BigData Cup.
  [IEEE Xplore](https://ieeexplore.ieee.org/document/10825254/)
- Arya et al., 2022. *RDD2022: A multi-national image dataset for automatic Road Damage
  Detection*. [arXiv:2209.08538](https://arxiv.org/abs/2209.08538)
- Kaya, Ö. & Çodur, M. Y., 2024. *N-RDD2024: Road damage and defects*.
  [doi:10.17632/27c8pwsd6v.3](https://doi.org/10.17632/27c8pwsd6v.3)
- Lin et al., 2017. *Focal Loss for Dense Object Detection*.
  [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Lin et al., 2014. *Microsoft COCO: Common Objects in Context* (mAP protocol).
  [arXiv:1405.0312](https://arxiv.org/abs/1405.0312)
- Everingham et al., 2010. *The PASCAL Visual Object Classes (VOC) Challenge* (AP@50
  protocol). [doi:10.1007/s11263-009-0275-4](https://doi.org/10.1007/s11263-009-0275-4)
