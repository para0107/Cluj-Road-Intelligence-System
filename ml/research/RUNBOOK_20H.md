# 20-hour weekend runbook

Exact commands, in order. Placeholders in `<ANGLE BRACKETS>` are the only things you
fill in.

**Time budget.** Steps 1–6 should take about 1.5–2.5 h, almost all of it the dataset
download. That leaves ~16 h of GPU time and 1.5 h reserved for getting results out.

---

## 0. Before the terminal — the one cross-account problem

You are on a **temporary** AWS account. Its credentials cannot write to your personal
S3 bucket, and everything here is deleted when the event ends.

So results leave by two routes, and you set both up now:

- **git** — metrics, configs, per-class AP, figures. All small text files, and the only
  things that actually matter for the research. This is the primary safety net.
- **personal S3** — model weights, which are large and which git should not carry.

Have ready: your repo URL, a Kaggle API token, and an access key for your **personal**
AWS account.

---

## 1. Open a terminal

Code Editor → Terminal → New Terminal.

```bash
nvidia-smi
```

Must print an **NVIDIA L4**. If it says "command not found", the space is on a CPU
instance — stop it, change the instance to `ml.g6.xlarge`, restart. Nothing below
works without this.

---

## 2. Clone and set up

```bash
cd ~
git clone <YOUR-REPO-URL> rdds
cd rdds

bash ml/aws/setup_env.sh
```

For a private repo, use a GitHub personal access token as the password.

Wait for `environment ready`. If it prints `setup incomplete`, fix the FAIL lines
first — starting a run on a broken environment wastes hours you cannot recover.

---

## 3. Credentials

**Kaggle** (for the dataset):

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"<KAGGLE_USERNAME>","key":"<KAGGLE_KEY>"}
EOF
chmod 600 ~/.kaggle/kaggle.json
```

Get those from Kaggle → Settings → API → Create New Token.

**Your personal AWS account** (for exporting weights):

```bash
aws configure --profile personal
# AWS Access Key ID     : <PERSONAL_KEY>
# AWS Secret Access Key : <PERSONAL_SECRET>
# Default region name   : us-west-2
# Default output format : json

aws sts get-caller-identity --profile personal
```

That last command must print your **personal** account id, not this temporary one.

> Do not use `aws sts get-caller-identity` without `--profile personal` to build a
> bucket name. On this account it returns the temporary account, and your export
> would go somewhere that is about to be deleted.

---

## 4. Dataset — inspect before you trust it

```bash
python ml/aws/fetch_kaggle.py --dataset nrdd2024 --inspect-only
```

Read the output. It must say:

```
ok   class names and ORDER match the canonical schema (10 classes)
```

If it says **BLOCKED**, stop and read the two orderings it prints. A reordered class
list means every label means something different from what the code assumes —
training would succeed and every number would be wrong.

Then stage it:

```bash
python ml/aws/fetch_kaggle.py --dataset nrdd2024 --stage /tmp/staged
```

**Read the duplicate count.** It tells you whether your existing 0.5637 baseline was
measured on leaked splits. Exit code 0 = clean, 2 = leakage found.

```bash
export DATA=/tmp/staged/dataset_nrdd2024_research.yaml
ls /tmp/staged            # expect train/ val/ test/ and the yaml
```

Skip RDD2022 this weekend. It only feeds E6 and would cost you an experiment.

---

## 5. Measure the GPU (~10 min, do not skip)

```bash
python ml/aws/weekend.py --calibrate --data $DATA
```

Note the **implied GPU_FACTOR**. My estimates assume 3.5. If it reports much less,
the plan has to shrink now rather than at hour 14.

---

## 6. Plan against the time actually left

```bash
python ml/aws/weekend.py --plan \
    --hours-left <HOURS FROM THE EVENT BANNER, MINUS 1.5> \
    --epochs 45 \
    --gpu-factor <FROM STEP 5> \
    --queue E0-baseline,E8-structural7,E3-800sq
```

Every row should say `fits: yes`. If not, drop `E3-800sq`, then reduce `--epochs` to
40. Do not reduce seeds below 2 — without them there is no noise floor and no
comparison can be judged.

---

## 7. Commit, then run

```bash
git add -A && git commit -m "research harness ready"
```

A dirty tree marks every run unreportable, so this is not optional.

```bash
tmux new -s rdds
```

Inside tmux:

```bash
cd ~/rdds
export DATA=/tmp/staged/dataset_nrdd2024_research.yaml

AWS_PROFILE=personal python ml/aws/weekend.py --run \
    --epochs 45 \
    --queue E0-baseline,E8-structural7,E3-800sq \
    --data $DATA \
    --export s3://<PERSONAL-BUCKET>/rdds-research \
    --hours-left <SAME AS STEP 6> \
    --gpu-factor <FROM STEP 5>
```

Detach with **Ctrl-B** then **D**. The run continues if your browser closes.

Reattach any time:

```bash
tmux attach -t rdds
```

`AWS_PROFILE=personal` matters: the only AWS call this makes is the export sync, and
it has to land in an account that still exists tomorrow.

---

## 8. While it runs — back up the small stuff to git

Every few hours, from a second terminal:

```bash
cd ~/rdds
git add -f runs/research/*/run.json runs/research/*/config.json \
             runs/research/*/metrics.csv runs/research/*/per_class_ap.json \
             runs/research/*/test_metrics.json
git commit -m "results $(date +%H:%M)" && git push
```

`-f` is needed because `.gitignore` excludes `runs/`. These files are a few KB each
and are the actual research output. Weights can be retrained; a lost `metrics.csv`
cannot be reconstructed.

---

## 9. Results

```bash
python ml/research/compare.py --runs runs/research
python ml/research/compare.py --runs runs/research \
    --baseline E0-baseline --challenger E8-structural7

python ml/research/anisotropy.py \
    --labels /tmp/staged/test/labels --images /tmp/staged/test/images \
    --per-class-ap runs/research/<E0-RUN-DIR>/per_class_ap.json \
    --out runs/research/E1_anisotropy

python ml/research/visualise.py --runs runs/research --out runs/research/_figures
```

The anisotropy result is the one that decides whether the E4 contribution is on for
the next weekend. Read its verdict before you plan anything else.

---

## 10. Before the clock runs out — at the 60-hour mark, or T-2h here

```bash
AWS_PROFILE=personal python ml/aws/weekend.py \
    --export-now s3://<PERSONAL-BUCKET>/rdds-research

git add -f runs/research/ && git commit -m "final results" && git push

AWS_PROFILE=personal aws s3 ls s3://<PERSONAL-BUCKET>/rdds-research/runs/
```

That last line is the one that matters. If it lists your run directories, the weekend
is safe. If it errors, fix it **now** — there is no recovery after the account expires.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `nvidia-smi: not found` | CPU space | Stop space, instance → `ml.g6.xlarge` |
| `torch.cuda.is_available()` False | CPU-only torch | `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121` |
| fetch_kaggle says BLOCKED | class order differs | Read both orderings. Do not `--force` unless you have checked manually |
| `stage_dataset` exit code 2 | leakage | Real finding. Note it — it means the old baseline was inflated |
| CUDA out of memory | batch too large | Add `--batch 8` to the trainer, or reduce `--epochs` |
| Run much slower than planned | GPU factor wrong | Re-run `--plan` with the corrected `--gpu-factor` and shorten the queue |
| Terminal died, run gone | no tmux | `tmux attach -t rdds`. If truly gone, restart from the last completed run |
