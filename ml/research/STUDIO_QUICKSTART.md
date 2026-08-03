# Running this from SageMaker Studio

Practical guide. The research reasoning is in `RESEARCH_PROGRAM.md`; this is the
operating manual.

---

## 0. The mental model

Three places, and it helps to keep them straight:

| Where | What it is | What runs there |
|---|---|---|
| **Studio** (your screenshot) | The control seat. Code Editor / JupyterLab / MLflow. | Editing, submitting jobs, reading results. **No training.** |
| **Training jobs** | GPU instances that spin up, train, shut down. | The actual training. You never SSH in. |
| **S3** | Storage. | The dataset, checkpoints, logs. |

You submit from Studio, AWS runs the GPU elsewhere, results come back to S3 and
MLflow. You pay only for the minutes the GPU exists.

**You do not have to use the CLI**, but you do need a terminal somewhere. The
cleanest option is Studio's Code Editor: credentials are already present, so nothing
has to be pasted or configured.

---

## 1. One-time setup

### 1.1 Open a terminal in Studio

From your Applications panel click **Code Editor** → create a space → open it. It is
VS Code in the browser. Terminal → New Terminal.

(JupyterLab works identically. Canvas is a no-code tool, not relevant here.)

### 1.2 Get the code in

```bash
git clone <your-repo-url> rdds
cd rdds
```

If the repo is private, a GitHub personal access token as the password works, or
push a bundle to S3 and `aws s3 cp` it down.

### 1.3 Install what the control seat needs

The training container installs its own dependencies. Studio only needs enough to
*submit* and *analyse*:

```bash
pip install sagemaker boto3 mlflow sagemaker-mlflow matplotlib pillow
```

### 1.4 Point at MLflow

You already have the app provisioned. Get its ARN:

```bash
aws sagemaker list-mlflow-tracking-servers --region us-west-2 \
  --query 'TrackingServerSummaries[].[TrackingServerName,TrackingServerArn]' --output table
```

Then, in every terminal you launch from:

```bash
export MLFLOW_TRACKING_URI=arn:aws:sagemaker:us-west-2:<account>:mlflow-tracking-server/<name>
export MLFLOW_EXPERIMENT=RDDS-detector
```

Put those two lines in `~/.bashrc` so they survive a restart. If you skip this,
everything still works — runs just write local artefacts only and the launcher says so.

### 1.5 Check credentials

Inside Studio this should already work:

```bash
aws sts get-caller-identity
```

**On your own laptop** you would set the four `AWS_*` variables instead. Those STS
credentials expire (usually within hours) and have to be refreshed. That alone is a
good reason to work from Studio.

---

## 2. Stage the dataset (do this once, before any GPU spend)

Download N-RDD2024 from the [Mendeley record](https://doi.org/10.17632/27c8pwsd6v.3)
— it needs terms accepted, so no script does it for you.

### 2.1 Dry run first

```bash
python ml/aws/stage_dataset.py --source /path/to/nrdd2024 --out /tmp/staged --dry-run
```

**Read the duplicate count.** If the source contains duplicates, your original
`train_oversampled/` vs `valid/` boundary may have leaked and the published
mAP50 0.5637 is inflated. That single number decides whether your baseline moves.

### 2.2 Build and upload

```bash
python ml/aws/stage_dataset.py \
    --source /path/to/nrdd2024 \
    --out /tmp/staged \
    --s3 s3://<your-bucket>/nrdd2024/v1
```

Exit code 0 means the splits are provably clean: no image, exact or near-duplicate,
appears in two splits. Exit code 2 means leakage was found and the splits are not
usable.

**Upload once.** Every class-set variant is derived inside the training container at
job start — you never upload a second copy.

### 2.3 RDD2022 (for the cross-dataset experiments)

Public, from [sekilab/RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector).
Annotations are PASCAL VOC XML; `ml/detection/data_prep/prep_rdd2022.py` already
handles the conversion. Then stage it the same way to `s3://<bucket>/rdd2022/v1`.

---

## 3. Submit experiments

### Always dry-run first

```bash
python ml/aws/launch.py --stage E8 --dry-run
```

Prints what would be submitted, the class set each uses, and a cost estimate.
Submits nothing.

### Submit

```bash
python ml/aws/launch.py --stage E8 \
    --data s3://<bucket>/nrdd2024/v1 \
    --output s3://<bucket>/rdds-research
```

Managed spot is on by default (roughly 70% cheaper, checkpointed so an interruption
costs one interval). Add `--on-demand` if you have a hard deadline.

### Useful selections

```bash
python ml/aws/launch.py --experiment E0-baseline --data … --output …   # one, all its seeds
python ml/aws/launch.py --stage E3 --data … --output …                 # a whole stage
python ml/aws/launch.py --all --dry-run                                # price everything
python ml/aws/launch.py --experiment E8-structural7 --seeds 1337,1338,1339 --data … --output …
python ml/aws/launch.py --status                                       # what's running
```

---

## 4. Watch it

Three views, in increasing order of usefulness:

- **Studio → Jobs → Training jobs** — status, instance, duration, and the log stream.
- **MLflow app** — live metric curves, run comparison, the parameter table. This is
  the one to keep open.
- **CloudWatch** — raw logs when a job fails and you need the traceback.

---

## 5. Analyse

Pull the artefacts down, then compare:

```bash
aws s3 sync s3://<bucket>/rdds-research/models ./runs/research

python ml/research/compare.py --runs runs/research
```

Head-to-head, which is where the actual judgement happens:

```bash
python ml/research/compare.py --runs runs/research \
    --baseline E8-all10 --challenger E8-structural7
```

And the hypothesis test that gates E4:

```bash
python ml/research/anisotropy.py \
    --labels /tmp/staged/test/labels --images /tmp/staged/test/images \
    --per-class-ap runs/research/<E0 run>/per_class_ap.json \
    --out runs/research/E1_anisotropy
```

---

## 6. Changing things — the four axes

All four are edits to `ml/research/experiments.py`. Nothing else needs touching, and
the launcher picks changes up automatically.

### Different parameters

```python
_reg(ExperimentSpec(
    id="E9-lowlr",
    stage="E9",
    title="Lower LR, longer schedule",
    hypothesis="The PSO learning rate was tuned at batch 4 on 4 GB and is too high at batch 16.",
    falsifier="mAP50-95 does not improve and the loss curve is unchanged.",
    gate="If this wins, the whole PSO result was batch-size-dependent and should be re-run.",
    overrides={"lr0": 1e-4, "weight_decay": 1e-3},
    epochs=100,
    seeds=(1337,),
))
```

`--check` will reject an override key Ultralytics does not accept, so a typo is caught
before it costs a job.

### Different class sets

Presets live in `ml/research/class_sets.py`:

```bash
python ml/research/class_sets.py                      # list them
python ml/research/class_sets.py --show structural7   # id mapping, what gets dropped
```

A new one is a few lines:

```python
_reg(ClassSet(
    name="potholes_only",
    description="Single-class pothole detector.",
    keep=["pothole"],
    rationale="Is the multi-class objective hurting the class that matters most?",
))
```

Then reference it with `class_set="potholes_only"` in a spec.

**Test it locally before submitting** — this costs nothing and catches mistakes:

```bash
python ml/research/class_sets.py --apply potholes_only --src /tmp/staged --dst /tmp/view
cat /tmp/view/dataset.yaml
cat /tmp/view/class_set.json     # box counts kept/dropped per split
```

### Different datasets

```bash
python ml/research/datasets.py                              # what's registered
python ml/research/datasets.py --transfer nrdd2024 rdd2022  # what is and isn't comparable
```

Then `dataset="rdd2022"` in a spec.

### Different models

`model="yolo12l.pt"` (or any Ultralytics id, or a path). The loader picks RT-DETR vs
YOLO from the name automatically.

---

## 7. Two traps

**Aggregate mAP across different class sets is meaningless.** Removing the hardest
classes raises the mean without improving a single prediction. `compare.py` detects
mismatched class counts, suppresses the aggregate, and judges on the shared classes
only. Do not work around this — it is the guard that stops you reporting a regression
as a win.

**A dirty git tree makes a run unreportable.** The metric then came from code that
isn't at the recorded SHA. Runs flag it, the leaderboard marks them ⚠dirty. Commit
before launching.

---

## 8. Suggested order

| Step | Command | Cost | Why |
|---|---|---|---|
| 1 | `stage_dataset.py --dry-run` | free | leakage check gates everything |
| 2 | `stage_dataset.py --s3 …` | pennies | one canonical upload |
| 3 | `launch.py --experiment E0-baseline` | ~$16 spot | real baseline + seed-noise floor |
| 4 | `anisotropy.py` | free | decides whether E4 is on |
| 5 | `launch.py --stage E8` | ~$27 spot | your class-selection question |
| 6 | `launch.py --stage E3` | ~$33 spot | resolution, the reliable win |
| 7 | `launch.py --stage E2` | ~$22 spot | is the baseline just old? |
| 8 | E4 | ~$50 spot | the contribution, if E1 supports it |

Roughly **$150 on spot** for the whole programme. Verify against real pricing and
your first run's actual epoch timings — the estimates come from a hardcoded table.

Steps 3 and 4 are the ones that matter most. Everything after them branches on what
they say.
