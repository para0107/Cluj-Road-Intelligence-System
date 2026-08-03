---
inclusion: fileMatch
fileMatchPattern: 'ml/**'
---

# Rules for the research harness (`ml/`)

`ml/` is the training and research layer. **The backend never imports it.** Inference
reads `ml/weights/` only.

`ml/detection/train.py` produced the current production checkpoint. **Do not edit it** —
the published numbers came from that file and it is kept for provenance.
`ml/detection/train_experiment.py` supersedes it for all research work.

## The current baseline, measured not assumed

RT-DETR-L on N-RDD2024 (10 classes), 57 epochs, converged:
val mAP50 **0.5637**, mAP50-95 **0.2945**, precision 0.655, recall **0.534**,
740 s/epoch on an RTX 2050.

Two things follow. Recall below precision is backwards for municipal survey, where a
missed defect costs more than a false alarm an operator dismisses in one click. And
every number above is a **validation** number selected over 57 epochs, so it is
optimistically biased by an unknown amount.

## Non-negotiables

**Every run goes through `ml/repro.py`.** Seed all four RNGs, capture the git SHA, save
the resolved config, log per-epoch metrics, fingerprint the dataset. A metric without
that provenance is not reportable.

**A dirty git tree makes a run unreportable.** The code that produced the number is not
the code at the recorded SHA. Commit first.

**The test split is touched exactly once**, at the end, after all model selection. Any
earlier peek recreates the selection bias the whole programme exists to remove.

**Never compare aggregate mAP across different class sets.** Removing the hardest
classes raises the mean without improving a single prediction.
`ml/research/compare.py` detects mismatched class counts, suppresses the aggregate, and
judges on shared classes only; `ml/research/visualise.py` mirrors that in the figures.
Do not work around either guard.

**A difference below the seed-noise floor is not a result.** E0 runs three seeds
specifically to measure that floor. `compare.py` refuses to call anything smaller a win.

**Class-set views are derived at job start, not uploaded.** One canonical dataset lives
in S3; `ml/research/class_sets.py` rewrites labels and symlinks images locally in
seconds. Never upload one dataset copy per ablation.

**Class ids must be remapped to contiguous 0..n-1 when classes are dropped.** YOLO
assumes `names` has no gaps. Keeping original ids after removing a middle class shifts
every later class by one, and the run still produces a plausible-looking number. This
is handled once, in `class_sets.py::source_to_output`.

## Adding an experiment

Edit `ml/research/experiments.py` only. Every spec carries its `hypothesis`,
`falsifier` and `gate` as data, so the run directory still says what the run was trying
to find out six weeks later. Then `python ml/research/experiments.py --check`, which
rejects override keys Ultralytics does not accept.

## Running

- **Personal account**: `ml/aws/launch.py` submits managed SageMaker training jobs.
- **Research-weekend account**: `ec2:RunInstances` is blocked and SageMaker AI classic
  may return a ValidationException, so use `ml/aws/weekend.py`, which runs experiments
  directly on the attached GPU against the 72-hour deadline and exports after every run.

## Reference points

N-RDD2024 (Kaya & Çodur, doi:10.17632/27c8pwsd6v.3) · RT-DETR (arXiv:2304.08069) ·
StripRFNet (arXiv:2510.16115) · RDD2022 (arXiv:2209.08538). ORDDC'2024's 86.18% F1 is
**not** a comparable target: it is 4-class RDD2022 with test-set pseudo-labelling.
