# Getting the datasets in

**Short answer: yes, you have to download them — but exactly once, and not onto your
laptop.** After the one-time staging into S3, every training job pulls from S3
automatically and you never touch the raw data again.

---

## The route that matters

```
Mendeley / GitHub  ──►  Studio space (temporary)  ──►  S3 (permanent)  ──►  training jobs
      manual              stage + verify              one canonical copy      automatic
```

Two things people get wrong here:

**Do not download to your laptop and upload from there.** You'd push gigabytes up a
home connection when the data can go straight into AWS at datacentre speed.

**Do not upload one copy per experiment.** Every class-set variant (7-class,
4-class, merged) is derived *inside the training container* at job start, in seconds.
One upload serves all of them.

---

## Storage: check this before you download anything

A Studio space has a fixed EBS volume, and the default is small (5 GB). N-RDD2024 and
especially RDD2022 will not fit alongside a staged copy.

```bash
df -h /home/sagemaker-user
```

You need roughly **2.5× the dataset size** free: the raw archive, the extracted copy,
and the staged output. If short:

- Studio → **Spaces** → your space → **Stop** → edit the EBS volume to 50–100 GB →
  restart. EBS is a few cents per GB-month, and you can shrink it afterwards.
- Or delete the raw archive right after extracting, which halves the requirement.

---

## N-RDD2024 (required — this is your training set)

Mendeley requires accepting terms in a browser, so no script fetches it for you.

**Option A — via your laptop's browser, then straight into Studio.** Download from
<https://doi.org/10.17632/27c8pwsd6v.3>, then drag the file into the Code Editor file
explorer. Fine for a few GB.

**Option B — Kaggle, and probably faster for you.** Your training yaml points at
`/kaggle/working/dataset`, so this data already exists as a Kaggle dataset under your
account. If so:

```bash
pip install kaggle
mkdir -p ~/.kaggle
# paste your kaggle.json (Kaggle → Account → Create New API Token) into ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d <your-username>/<dataset-slug> -p /tmp/nrdd --unzip
```

This downloads directly into Studio at AWS network speed. It's the fastest path if
the Kaggle copy still exists.

**Then stage it:**

```bash
cd ~/Cluj-Road-Intelligence-System

# 1. Dry run. Costs nothing, writes nothing, and answers the question that
#    determines whether your existing 0.5637 baseline is trustworthy.
python ml/aws/stage_dataset.py --source /tmp/nrdd --out /tmp/staged --dry-run
```

**Read the duplicate line.** If the source contains duplicates and the original
`train_oversampled/` vs `valid/` split was made without grouping them, images leaked
across the boundary and the published mAP is inflated. That one number decides
whether your baseline moves before you spend anything on GPU.

```bash
# 2. Build the splits and upload
python ml/aws/stage_dataset.py \
    --source /tmp/nrdd \
    --out /tmp/staged \
    --s3 s3://<your-bucket>/nrdd2024/v1
```

Exit code 0 means provably clean splits. Exit code 2 means leakage was found and the
splits are unusable — fix before training.

Create the bucket first if you need one:

```bash
aws s3 mb s3://rdds-research-$(aws sts get-caller-identity --query Account --output text) \
    --region us-west-2
```

---

## RDD2022 (optional — only for the cross-dataset experiments)

Public, no terms gate, but large (~47,000 images across six countries).

```bash
# Country archives are separate; start with Japan + Czech unless you need all six
wget https://github.com/sekilab/RoadDamageDetector/... -P /tmp/rdd2022
```

See <https://github.com/sekilab/RoadDamageDetector> for the current download links —
they move between releases, so check rather than trusting a copied URL.

Annotations are PASCAL VOC XML, not YOLO. Your repo already has the converter, but
note `scripts/` is gitignored and therefore **not in your Studio clone**:
`ml/detection/data_prep/prep_rdd2022.py` is the one that is present.

```bash
python ml/detection/data_prep/prep_rdd2022.py   # check its paths first
python ml/aws/stage_dataset.py --source /tmp/rdd2022_yolo --out /tmp/staged_rdd \
    --s3 s3://<bucket>/rdd2022/v1
```

**Skip this until E0 and E8 are done.** It only feeds E6, which is late in the ladder,
and it is the largest download in the programme.

---

## Not needed

| Thing | Why not |
|---|---|
| `ml/weights/best.pt` | Gitignored, and training jobs start from Ultralytics' pretrained COCO weights, downloaded automatically inside the container. |
| `rtdetr-l.pt` | Same — fetched by Ultralytics on first use. |
| `data/` | Gitignored. Session outputs from your local pipeline; irrelevant to training. |
| `runs/` | Gitignored. Historical results stay on your local machine. |
| `scripts/` | Gitignored, so it isn't in your Studio clone. Nothing in the research programme needs it. |

---

## If the dataset is too big for the space

Stage it with a SageMaker Processing job instead of in the Code Editor — the data
never touches your space at all:

```python
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput

ScriptProcessor(
    image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.3.0-cpu-py311",
    command=["python3"], role=role,
    instance_type="ml.m5.2xlarge", instance_count=1,
    volume_size_in_gb=200,
).run(
    code="ml/aws/stage_dataset.py",
    inputs=[ProcessingInput(source="s3://<bucket>/raw/nrdd2024",
                            destination="/opt/ml/processing/input")],
    outputs=[ProcessingOutput(source="/opt/ml/processing/output",
                              destination="s3://<bucket>/nrdd2024/v1")],
    arguments=["--source", "/opt/ml/processing/input",
               "--out", "/opt/ml/processing/output"],
)
```

Worth it for RDD2022. Overkill for N-RDD2024 unless your space is small.

---

## Checklist before your first training job

- [ ] `df -h` shows enough room
- [ ] N-RDD2024 extracted somewhere in the space
- [ ] `stage_dataset.py --dry-run` run, and **the duplicate count read**
- [ ] Staged and uploaded, exit code 0
- [ ] `aws s3 ls s3://<bucket>/nrdd2024/v1/` shows train/, val/, test/
- [ ] `git commit` — a dirty tree makes every run unreportable
