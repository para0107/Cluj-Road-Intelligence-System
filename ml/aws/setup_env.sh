#!/usr/bin/env bash
# ml/aws/setup_env.sh
# ---------------------------------------------------------------------------
# Bootstrap a fresh SageMaker Unified Studio GPU instance for the RDDS research
# programme. Run once after the compute environment starts.
#
#   bash ml/aws/setup_env.sh
#
# The participant guide stops at "wait 2-3 minutes for the instance to start".
# This picks up from there.
#
# Design notes:
#
#   TORCH IS NOT REINSTALLED IF IT ALREADY WORKS. The SageMaker images ship a
#   CUDA-enabled torch. `pip install ultralytics` would happily pull a different
#   torch build over it - a multi-GB download that can land a CPU-only wheel and
#   silently cost you the GPU. So torch is checked first and protected.
#
#   IDEMPOTENT. Safe to re-run after a disconnect; it skips whatever is already
#   present rather than starting over.
#
#   FAILS LOUDLY, NOT SILENTLY. On a 72-hour account a setup that half-worked and
#   said nothing is worse than one that stops and tells you.
# ---------------------------------------------------------------------------
set -uo pipefail

# --check  : verify the environment, install nothing. Fast. Use it to confirm a
#            setup afterwards, or to see what is missing before committing to a
#            multi-GB download on a clock that is running.
CHECK_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --check|--check-only|-n) CHECK_ONLY=1 ;;
        -h|--help)
            echo "usage: bash ml/aws/setup_env.sh [--check]"
            echo "  --check   verify only, install nothing"
            exit 0 ;;
    esac
done

BOLD=$'\033[1m'; RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; RST=$'\033[0m'
say()  { printf "%s\n" "${BOLD}==> $*${RST}"; }
ok()   { printf "%s\n" "  ${GRN}ok${RST}   $*"; }
warn() { printf "%s\n" "  ${YLW}warn${RST} $*"; }
bad()  { printf "%s\n" "  ${RED}FAIL${RST} $*"; }
skip() { printf "%s\n" "  ${YLW}skip${RST} $* (--check)"; }

FAILED=0
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || { echo "cannot cd to repo root"; exit 1; }
say "repo: $REPO_ROOT"

# ---------------------------------------------------------------------------
say "1/6  hardware"
# ---------------------------------------------------------------------------
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$GPU_NAME" ]; then
        ok "GPU: $GPU_NAME ($GPU_MEM)"
    else
        bad "nvidia-smi present but reported no GPU"; FAILED=1
    fi
else
    bad "no nvidia-smi. This is NOT a GPU instance."
    echo "     In Unified Studio: Compute Environment -> select sc.g6.xlarge"
    echo "     (L4 24GB, the guide's pick for training) -> Apply."
    FAILED=1
fi

DISK_AVAIL_G=$(df -BG --output=avail "$HOME" 2>/dev/null | tail -1 | tr -dc '0-9')
DISK_AVAIL_G=${DISK_AVAIL_G:-0}
if [ "$DISK_AVAIL_G" -ge 40 ]; then
    ok "disk: ${DISK_AVAIL_G} GB free"
elif [ "$DISK_AVAIL_G" -ge 15 ]; then
    warn "disk: only ${DISK_AVAIL_G} GB free. Enough for N-RDD2024 if you delete the"
    echo "       raw archive after extracting. Not enough for RDD2022."
else
    bad "disk: ${DISK_AVAIL_G} GB free - too little for a dataset plus staging."
    echo "     Stop the space, raise Storage toward the 100 GB maximum, restart."
    FAILED=1
fi

# ---------------------------------------------------------------------------
say "2/6  python and torch"
# ---------------------------------------------------------------------------
PY=$(command -v python3 || command -v python)
[ -z "$PY" ] && { bad "no python found"; exit 1; }
ok "python: $($PY --version 2>&1)"

TORCH_STATUS=$($PY - <<'EOF' 2>/dev/null
try:
    import torch
    print(f"{torch.__version__}|{torch.cuda.is_available()}|{torch.version.cuda}")
except ImportError:
    print("MISSING||")
EOF
)
TORCH_VER="${TORCH_STATUS%%|*}"
TORCH_CUDA=$(echo "$TORCH_STATUS" | cut -d'|' -f2)

# The stack weekend 1's baseline was produced on, read from
# runs/research/*/run.json: torch 2.8.0, CUDA 12.9, ultralytics 8.4.115.
#
# Both parts of this matter.
#   - cu121 has NO torch 2.8 wheel. It was dropped after the 2.4/2.5 line, so an
#     unpinned install from that index silently DOWNGRADES torch to ~2.4.
#   - `--force-reinstall torch` with no version takes the newest wheel on the index
#     (2.13 at the time of writing), which is a different numerical stack from the
#     one the 0.1991 baseline came from. Comparisons across it are not clean.
# Pin both, so a re-staged run is comparable to weekend 1 by construction.
TORCH_PIN="torch==2.8.0"
TV_PIN="torchvision==0.23.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu129"
TORCH_FIX="pip install --force-reinstall --no-cache-dir $TORCH_PIN $TV_PIN --index-url $TORCH_INDEX"

if [ "$TORCH_VER" = "MISSING" ]; then
    if [ "$CHECK_ONLY" -eq 1 ]; then
        skip "torch missing, would install $TORCH_PIN (cu129)"; FAILED=1
    else
        warn "torch not installed - installing $TORCH_PIN cu129 (several GB, be patient)"
        $PY -m pip install --quiet $TORCH_PIN $TV_PIN --index-url "$TORCH_INDEX" \
            || { bad "torch install failed"; FAILED=1; }
    fi
elif [ "$TORCH_CUDA" = "True" ]; then
    ok "torch $TORCH_VER with CUDA - leaving it alone"
    case "$TORCH_VER" in
        2.8.*) : ;;
        *) warn "weekend 1 ran torch 2.8.0; this is $TORCH_VER. Fine to proceed, but "
           warn "record it - a different numerical stack weakens the comparison." ;;
    esac
else
    bad "torch $TORCH_VER is installed but CUDA is NOT available"
    echo "     A CPU-only torch on a GPU box means training silently runs ~50x slower."
    echo "     Fix (pinned to weekend 1's stack; cu121 has no torch 2.8 wheel):"
    echo "       $TORCH_FIX"
    FAILED=1
fi

# ---------------------------------------------------------------------------
say "3/6  project dependencies"
# ---------------------------------------------------------------------------
# ultralytics declares torch as a dependency. --no-deps keeps pip from replacing a
# working CUDA torch with whatever it prefers; the remaining deps are listed
# explicitly below.
need_pkg() { $PY -c "import $1" >/dev/null 2>&1; }

if need_pkg ultralytics; then
    ok "ultralytics $($PY -c 'import ultralytics;print(ultralytics.__version__)' 2>/dev/null)"
elif [ "$CHECK_ONLY" -eq 1 ]; then
    skip "ultralytics missing, would install"; FAILED=1
else
    say "    installing ultralytics (--no-deps, to protect torch)"
    $PY -m pip install --quiet --no-deps ultralytics || { bad "ultralytics failed"; FAILED=1; }
fi

# Ultralytics' real runtime deps, minus torch/torchvision which we handled above.
# opencv-python-headless rather than opencv-python: no display on a server, and the
# GUI build pulls X11 libraries that are not present.
# numpy is PINNED. Unpinned, pip upgrades it past what the preinstalled torch,
# OpenCV, numba and sagemaker-studio builds support. Those all cross the C
# extension boundary with numpy, so a mismatch shows up as a segfault or a strange
# dtype error partway through the first epoch rather than at install time.
# <2.3 satisfies sagemaker-studio (<2.3.0) and numba (<2.5) simultaneously.
PIP_PKGS=(
    "numpy<2.3"
    "opencv-python-headless" "pillow" "pyyaml" "requests" "scipy"
    "matplotlib" "pandas" "tqdm" "psutil" "py-cpuinfo" "ultralytics-thop"
    "boto3" "mlflow" "sagemaker-mlflow"
)
IMPORT_NAMES=(cv2 PIL yaml requests scipy matplotlib pandas tqdm psutil cpuinfo thop boto3 mlflow)
if [ "$CHECK_ONLY" -eq 1 ]; then
    MISSING=()
    for m in "${IMPORT_NAMES[@]}"; do need_pkg "$m" || MISSING+=("$m"); done
    if [ ${#MISSING[@]} -eq 0 ]; then
        ok "support packages present"
    else
        skip "would install: ${MISSING[*]}"
    fi
else
    say "    installing support packages"
    $PY -m pip install --quiet "${PIP_PKGS[@]}" 2>&1 | grep -Ev "^\s*$" | tail -3
    ok "support packages done"
fi

# ---------------------------------------------------------------------------
say "4/6  environment variables"
# ---------------------------------------------------------------------------
# Ultralytics and matplotlib write config into $HOME by default, which is not always
# writable in a managed container. Point both at /tmp before anything imports them.
BASHRC="$HOME/.bashrc"
add_env() {
    grep -qF "$1" "$BASHRC" 2>/dev/null || echo "$1" >> "$BASHRC"
    eval "export $1"
}
add_env "YOLO_CONFIG_DIR=/tmp/ultralytics"
add_env "MPLCONFIGDIR=/tmp/matplotlib"
mkdir -p /tmp/ultralytics /tmp/matplotlib
ok "YOLO_CONFIG_DIR and MPLCONFIGDIR set (and persisted to ~/.bashrc)"

if [ -n "${MLFLOW_TRACKING_URI:-}" ]; then
    ok "MLFLOW_TRACKING_URI is set"
else
    warn "MLFLOW_TRACKING_URI not set - runs write local artefacts only (that is fine)"
fi

# ---------------------------------------------------------------------------
say "5/6  verifying the research harness imports"
# ---------------------------------------------------------------------------
$PY - <<'EOF'
import sys
sys.path.insert(0, ".")
mods = [
    "ml.repro", "ml.tracking",
    "ml.research.experiments", "ml.research.class_sets",
    "ml.research.datasets", "ml.research.compare",
    "ml.research.anisotropy", "ml.research.visualise",
    "ml.aws.stage_dataset", "ml.aws.weekend",
]
bad = []
for m in mods:
    try:
        __import__(m)
    except Exception as exc:
        bad.append(f"{m}: {exc}")
if bad:
    print("  FAIL import errors:")
    for b in bad:
        print("    " + b)
    sys.exit(1)
from ml.research.experiments import REGISTRY, check_registry
problems = check_registry()
if problems:
    print("  FAIL registry problems:")
    for p in problems:
        print("    " + p)
    sys.exit(1)
print(f"  ok   all {len(mods)} modules import; registry clean "
      f"({len(REGISTRY)} experiments)")
EOF
[ $? -ne 0 ] && FAILED=1

# ---------------------------------------------------------------------------
say "6/6  git and credentials"
# ---------------------------------------------------------------------------
if git rev-parse --git-dir >/dev/null 2>&1; then
    if [ -z "$(git status --porcelain)" ]; then
        ok "git tree is clean - runs will be reportable"
    else
        warn "git tree is DIRTY. Runs started now are flagged unreportable"
        echo "       (the code would not match the recorded SHA). Commit first."
    fi
else
    warn "not a git repo - runs cannot record a commit SHA"
fi

if command -v aws >/dev/null 2>&1; then
    ACCT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
    if [ -n "$ACCT" ] && [ "$ACCT" != "None" ]; then
        ok "aws CLI authenticated (account $ACCT)"
        echo "       NOTE: on a research weekend this is the TEMPORARY account."
        echo "       Your --export bucket must be in your PERSONAL account."
    else
        warn "aws CLI present but no valid credentials - export will fail"
    fi
else
    bad "aws CLI missing - you cannot export results before the account expires"
    FAILED=1
fi

# ---------------------------------------------------------------------------
echo
if [ "$FAILED" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    say "7/7  smoke test (proves the stack actually works, not just imports)"
    if $PY ml/aws/smoke_test.py --quick; then
        ok "stack verified"
    else
        bad "smoke test failed - see the guidance above"
        FAILED=1
    fi
    echo
fi

if [ "$FAILED" -eq 0 ]; then
    say "${GRN}environment ready${RST}"
    cat <<'NEXT'

Next, in order:

  1. Stage the dataset (dry run first - it tells you whether the existing
     baseline leaked, which decides whether your 0.5637 still means anything):
       python ml/aws/stage_dataset.py --source <extracted> --out /tmp/staged --dry-run
       python ml/aws/stage_dataset.py --source <extracted> --out /tmp/staged

  2. Measure this GPU instead of assuming its speed (~10 min):
       python ml/aws/weekend.py --calibrate \
           --data /tmp/staged/dataset_nrdd2024_research.yaml

  3. Plan against the real deadline:
       python ml/aws/weekend.py --plan --deadline "YYYY-MM-DD HH:MM" \
           --gpu-factor <from step 2>

  4. Commit, then run with continuous export to your PERSONAL bucket:
       git commit -am "research harness"
       python ml/aws/weekend.py --run \
           --queue E0-baseline,E8-all10,E8-structural7,E3-800sq,E3-1024x576 \
           --data /tmp/staged/dataset_nrdd2024_research.yaml \
           --export s3://<PERSONAL-BUCKET>/rdds-research \
           --deadline "YYYY-MM-DD HH:MM"

Set an alarm at the 60-hour mark. Everything here is deleted at 72 hours with
no recovery.
NEXT
    exit 0
else
    say "${RED}setup incomplete${RST} - fix the FAIL lines above before training."
    echo "Starting a run on a broken environment wastes hours you cannot get back."
    exit 1
fi
