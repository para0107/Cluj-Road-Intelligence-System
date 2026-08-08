# Weekend 2 — 60-hour runbook

Exact commands, in order. Only `<ANGLE BRACKETS>` need filling in.

The reasoning behind the queue is in `WEEKEND2_PLAN.md`. This file is just the sequence.

**Time budget.** Parts 0–4 take about 2.5–3.5 h, most of it the dataset download and
staging. That leaves ~52 h of GPU and 1.5 h reserved for getting results out.

---

## Part 0 — Pick the right instance

**`ml.m5.16xlarge` will not work.** The M5 family is general-purpose CPU: 64 vCPU,
256 GiB RAM, **0 GPU**. `nvidia-smi` fails, `torch.cuda.is_available()` is False, and
`setup_env.sh` reports a broken environment. It is also **$3.686/hr**, which is more than
three times the GPU instance you actually want.

| Instance | GPU | VRAM | $/hr | 60 h | Verdict |
|---|---|---|---:|---:|---|
| `ml.m5.16xlarge` | **none** | — | 3.686 | $221 | cannot train, costs the most |
| `ml.g6.xlarge` | L4 | 24 GB | 1.127 | $68 | **use this** — weekend 1's instance, all timings assume it |
| `ml.g4dn.xlarge` | T4 | 16 GB | 0.736 | $44 | cheaper, ~2× slower, tight VRAM at 1024 px |

Prices are us-east-1 on-demand as of 8 Aug 2026; check your own region before committing.

`ml.g6.xlarge` is both cheaper and the only option for which the plan's hour figures hold,
since every timing in `WEEKEND2_PLAN.md` was measured on an L4. Switching to it *saves*
about $150 over the window.

---

## Part 0b — On your Windows machine, before you touch AWS

`.gitignore` has been rewritten so the harness reaches GitHub. `ml/aws` and `ml/research`
used to be blanket-ignored, and commit `55e2f3e` removed them from HEAD, so a fresh clone
contained none of the code this runbook invokes. They are now allowlists: `.py`, `.sh` and
research `.md` are tracked; artefacts, papers and caches stay out.

```cmd
cd C:\Facultate\pothole-detection\Pothole-Detection
del .git\index.lock

git add .gitignore ml/aws ml/research ml/detection/train_experiment.py ml/repro.py
git add EXPERIMENTS.md
git commit -m "weekend 2: track research harness, add geo splits + E9/E10"
git push
```

No `-f` needed any more for the code. Run artefacts still are ignored on purpose, so
those keep needing it:

```cmd
git add -f runs/research/
git commit -m "weekend 1 artefacts"
git push
```

Now prove it worked, because this is the failure that costs a morning:

```cmd
git clone <YOUR-REPO-URL> C:\temp\clonecheck
dir C:\temp\clonecheck\ml\research
dir C:\temp\clonecheck\ml\aws
```

You must see `experiments.py`, `geo_splits.py`, `class_sets.py`, `compare.py` and
`stage_all.py`. If any are missing, the push did not take.

---

## Part 1 — Studio instance

Code Editor → Terminal → New Terminal.

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print **NVIDIA L4** and `True`. If `nvidia-smi` says command not found, the space is
on a CPU instance (see Part 0) — stop it, change the instance to `ml.g6.xlarge`, restart.
Nothing below works without this, and no amount of vCPU substitutes for it.

```bash
cd ~
git clone <YOUR-REPO-URL> rdds
cd rdds
bash ml/aws/setup_env.sh
```

Wait for `environment ready`. Fix any FAIL lines before continuing.

```bash
ls ml/research/geo_splits.py ml/aws/stage_all.py   # both must exist
python ml/research/geo_splits.py --self-test        # 26 checks, all must pass
python ml/research/experiments.py --check           # "Registry clean: 31 experiments"
```

---

## Part 2 — Credentials

```bash
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json <<'EOF'
{"username":"<KAGGLE_USERNAME>","key":"<KAGGLE_KEY>"}
EOF
chmod 600 ~/.kaggle/kaggle.json

aws configure --profile personal
# region us-west-2, output json
aws sts get-caller-identity --profile personal
```

That last command must print your **personal** account id, not the temporary one. Results
leave via your personal bucket; the event account is destroyed with no recovery.

---

## Part 3 — Data

### 3a. Fetch

```bash
python ml/aws/fetch_kaggle.py --dataset nrdd2024 --inspect-only
```

Must say `class names and ORDER match the canonical schema (10 classes)`. If BLOCKED,
stop and read both orderings — a reordered class list makes every label mean something
different and training will succeed while every number is wrong.

```bash
python ml/aws/fetch_kaggle.py --dataset nrdd2024 --stage /tmp/src
export SRC=/tmp/src
```

### 3b. Confirm the country composition

```bash
python ml/research/geo_splits.py --source $SRC
```

Expected, matching weekend 1's manifest:

```
  japan       7198   37.9%
  usa         4804   25.3%
  norway      2803   14.8%
  china       1977   10.4%
  india       1221    6.4%
  czech        992    5.2%
  TOTAL      18995
```

If any images come back `unknown`, stop — a LOCO fold built over unlabelled images does
not measure domain shift. Extend `geo_splits.COUNTRIES`.

If these counts differ from the above, **update the control sizes** in
`DATA_VARIANTS` (`control_2803`, `control_1221`, `control_992`) to match, or each control
will hold out the wrong number and stop being a control.

### 3c. Gate 2 — is any near-duplicate threshold trustworthy? (~10 min)

```bash
python ml/aws/stage_dataset.py --source $SRC --calibrate-hash 300
echo "exit=$?"
```

- **exit 0** (`separated: true`) → note the `recommended_threshold`, and add
  `--hash dhash --hash-threshold <N>` to the `stage_all.py` call in 3d.
- **exit 2** (`separated: false`) → change nothing. `--hash none` is the default and is
  the right answer. Record it as a stated limitation.

Do not skip this and do not guess. Weekend 1's aHash reported 72% of the dataset as
duplicates and that number was a bug, not a finding.

### 3d. Build every split the queue needs

One command builds all seven variant directories with names that match what the runner
looks for:

```bash
cat > /tmp/queue.txt <<'EOF'
E0-baseline
E9-oversamplenone
E9-oversample60
E8-structural7
E10-loco-norway
E10-control-norway
E10-loco-india
E10-control-india
E3-1024x576
EOF

python ml/aws/stage_all.py --source $SRC --out /tmp/data --queue-file /tmp/queue.txt
echo "exit=$?"
```

Exit **0** = every variant staged and every leakage check clean. Exit 1 or 2 = stop and
read the summary table; do not train on a split that reported LEAKAGE.

```bash
export DATA_ROOT=/tmp/data
export DATA=$DATA_ROOT/staged_standard/dataset_nrdd2024_research.yaml
ls $DATA_ROOT     # staged_standard, staged_no_oversample, staged_loco_norway, ...
```

**What goes where, in one picture:**

```
~/rdds/                                     the clone; run everything from here
~/rdds/runs/research/                       every result lands here
$SRC       = /tmp/src                       raw N-RDD2024 from Kaggle
$DATA_ROOT = /tmp/data                      all staged variants
  staged_standard/                            E0, E8, E3   <- also $DATA
  staged_no_oversample/                       E9-oversamplenone
  staged_oversample_60/                       E9-oversample60
  staged_loco_norway/  staged_control_2803/   E10 norway pair
  staged_loco_india/   staged_control_1221/   E10 india pair
  _staging_report.json
```

`weekend.py --data-root $DATA_ROOT` resolves each experiment to its own split. Without
that flag, any E9/E10 run is **skipped with a loud error** rather than silently trained
on the wrong data.

---

## Part 4 — Gate 1: batch size (~30 min GPU)

`resolve_batch` returns 16 on this card, which uses under a quarter of it. This single
measurement moves the whole queue by roughly 15 hours.

```bash
python - <<'EOF'
import time, torch
from ultralytics import RTDETR
import os
DATA = os.environ["DATA"]
for b in (16, 32, 48, 64):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        t = time.time()
        RTDETR("rtdetr-l.pt").train(data=DATA, epochs=1, batch=b, imgsz=640,
                                    device=0, amp=True, val=False,
                                    project="/tmp/batchcal", name=f"b{b}", exist_ok=True)
        print(f"batch {b:3d}  {time.time()-t:6.0f}s  "
              f"{torch.cuda.max_memory_allocated()/2**30:4.1f} GB")
    except torch.cuda.OutOfMemoryError:
        print(f"batch {b:3d}  OOM"); break
EOF
```

Divide the batch-16 time by the best time to get the throughput gain.

| Gain | What to do |
|---|---|
| **≥ 1.25×** | Adopt the batch. Add `--batch <N>` to the run command. Keep all three E0 seeds — the batch change means the noise floor must be re-measured at the new setting. |
| **< 1.25×** | Keep batch 16. Cut the queue to **one** clean E0 seed; weekend 1's floor (0.0039) still applies and the clean run only clears the ⚠dirty flag. Saves ~5 h. |

---

## Part 5 — Commit clean, then run

A dirty tree marks every run unreportable. This is the single thing weekend 1 got wrong.

```bash
git add -A && git commit -m "weekend 2 harness ready" && git push
git status --porcelain      # must print NOTHING
```

```bash
tmux new -s rdds
```

Inside tmux:

```bash
cd ~/rdds
export SRC=/tmp/src DATA_ROOT=/tmp/data
export DATA=$DATA_ROOT/staged_standard/dataset_nrdd2024_research.yaml

python ml/aws/weekend.py --plan --hours-left <HOURS FROM BANNER MINUS 1.5> \
    --epochs 20 --queue "$(paste -sd, /tmp/queue.txt)"
```

Every row must say `fits: yes`. If not, drop from the bottom in this order:
`E3-1024x576` → `E10-control-india` + `E10-loco-india` → third E0 seed.
Never drop: two E0 seeds, the norway LOCO **pair**, or a control without its fold.

Then run it:

```bash
AWS_PROFILE=personal python ml/aws/weekend.py --run \
    --epochs 20 \
    --queue "$(paste -sd, /tmp/queue.txt)" \
    --data $DATA \
    --data-root $DATA_ROOT \
    --export s3://<PERSONAL-BUCKET>/rdds-research \
    --hours-left <SAME AS ABOVE>
```

Detach with **Ctrl-B** then **D**. Reattach with `tmux attach -t rdds`.

Each run prints the variant it resolved:

```
[run] E10-loco-norway seed=1337  ~2.2 h
[data] variant=loco_norway
       /tmp/data/staged_loco_norway/dataset_nrdd2024_research.yaml
```

**Check that line on the first E9/E10 run.** If it says `variant=standard` for one of
them, `--data-root` did not take and the run is measuring nothing.

---

## Part 6 — While it runs

From a second terminal, every few hours:

```bash
cd ~/rdds
git add -f runs/research/*/run.json runs/research/*/config.json \
           runs/research/*/metrics.csv runs/research/*/per_class_ap.json \
           runs/research/*/test_metrics.json runs/research/_weekend_log.json
git commit -m "results $(date +%H:%M)" && git push
```

A few KB each, and they are the actual research output. This is the safety net that
worked last time.

---

## Part 7 — The 60-epoch winner (start by T-14 h)

Once the 20-epoch sweep is done, pick the best configuration and train it to convergence.
This is the only converged number the weekend produces and the only one to quote as an
absolute figure.

```bash
python ml/research/compare.py --runs runs/research      # read the leaderboard first

AWS_PROFILE=personal python ml/aws/weekend.py --run \
    --epochs 60 --queue <WINNING-EXPERIMENT-ID> \
    --data $DATA --data-root $DATA_ROOT \
    --export s3://<PERSONAL-BUCKET>/rdds-research \
    --hours-left <WHAT IS LEFT>
```

Budget 12.5 h at 1024×576, 8.7 h at 640 (less if Gate 1 gave a batch win). Do not start
it without that much left — the scheduler will refuse anyway.

---

## Part 8 — Analysis

```bash
python ml/research/compare.py --runs runs/research

# E8: shared classes only, aggregates suppressed automatically
python ml/research/compare.py --runs runs/research \
    --baseline E0-baseline --challenger E8-structural7

# E10: the fold against ITS CONTROL. Never quote the fold alone.
python ml/research/compare.py --runs runs/research \
    --baseline E10-control-norway --challenger E10-loco-norway
python ml/research/compare.py --runs runs/research \
    --baseline E10-control-india --challenger E10-loco-india

# E9: read per-class AP for the rare classes, not the aggregate
python ml/research/dump_results.py --runs runs/research --full

python ml/research/visualise.py --runs runs/research --out runs/research/_figures
```

`domain_shift_effect = AP(control) − AP(loco)`. That difference is the E10 result. The
LOCO number on its own measures training-set size.

**E5 is free and worth doing here** — per-class threshold sweep on *val*, evaluated once
on test, reporting F2 and false-positives-per-kilometre. No training run needed, and
recall 0.438 vs precision 0.615 is the confirmed operating problem.

---

## Part 9 — Before the clock runs out (T-2 h)

```bash
AWS_PROFILE=personal python ml/aws/weekend.py \
    --export-now s3://<PERSONAL-BUCKET>/rdds-research

git add -f runs/research/ && git commit -m "final results" && git push

AWS_PROFILE=personal aws s3 ls s3://<PERSONAL-BUCKET>/rdds-research/runs/
```

That last line is the one that matters. If it lists your run directories, the weekend is
safe. If it errors, fix it **now** — there is no recovery after the account expires.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `ml/research/geo_splits.py` missing after clone | Part 0b not done | commit `.gitignore` + `ml/aws` + `ml/research`, push, re-clone |
| `nvidia-smi: not found` | CPU instance (m5, c5, t3) | Stop space, instance → `ml.g6.xlarge`. See Part 0 |
| `torch.cuda.is_available()` False on a g-instance | CPU-only torch wheel | `pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121` |
| `[skip] E10-... needs data_variant` | `--data-root` not passed | add `--data-root $DATA_ROOT` |
| `[data] variant=standard` on an E9/E10 run | same | stop the queue; those runs measure nothing |
| `stage_all.py` exit 2 | leakage in a variant | read the summary table; do not train on it |
| `[dedupe] ABORT: absorbed N%` | hash collapsed | this is the guard working. Use `--hash none` |
| `n_test_images=2803 but only N images exist` | control size exceeds the archive | the source is short; re-check Part 3b counts |
| CUDA OOM | batch too large | lower `--batch`; Gate 1 should have caught it |
| Run much slower than planned | pixel-scaling estimate off | re-run `--plan`, drop from the bottom of Part 5 |
| Terminal died | no tmux | `tmux attach -t rdds` |
| `kill` did not stop training | orphaned child | `pkill -f train_experiment.py` too, or it keeps the GPU |
