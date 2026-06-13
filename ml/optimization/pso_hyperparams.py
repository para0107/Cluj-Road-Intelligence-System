"""
ml/optimization/pso_hyperparams.py

PURPOSE:
    Uses Particle Swarm Optimization (PSO) to find optimal hyperparameters
    for RT-DETR-L training. Each particle represents a candidate set of
    hyperparameters; fitness = validation mAP50-95 after N quick epochs.

    After PSO completes, writes ml/optimization/pso_best.json.
    train.py loads this file automatically on the next run.

ALGORITHM:
    Standard PSO with inertia weight decay:
        v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x(t+1) = x(t) + v(t+1)

    w  = inertia (decays 0.9 → 0.4 over iterations)
    c1 = cognitive coefficient (personal best attraction) = 1.5
    c2 = social coefficient   (global best attraction)   = 1.5

SEARCH SPACE (7 dimensions):
    lr0           [1e-5, 5e-4]   initial learning rate
    weight_decay  [1e-5, 1e-3]   L2 regularisation
    warmup_epochs [1,    5   ]   warmup duration
    mosaic        [0.5,  1.0 ]   mosaic augmentation probability
    mixup         [0.0,  0.3 ]   mixup alpha
    box           [5.0,  10.0]   box loss weight
    cls           [0.3,  1.0 ]   classification loss weight

FITNESS FUNCTION:
    Train RT-DETR-L for `eval_epochs` epochs (default: 5) on the full
    training set, evaluate on val set, return mAP50-95.

    5 epochs per particle × 15 particles × 10 iterations = 750 epochs total
    ≈ 9-12 hours on RTX 2050. Run overnight after baseline training.

USAGE:
    # Standard PSO (15 particles, 10 iterations, 5 eval epochs each)
    python ml/optimization/pso_hyperparams.py

    # Quick test (fewer particles/iterations)
    python ml/optimization/pso_hyperparams.py --particles 5 --iterations 3 --eval_epochs 2

    # Resume from checkpoint if interrupted
    python ml/optimization/pso_hyperparams.py --resume

INPUT:
    data/detection/dataset.yaml          (must already be correct)
    data/detection/train_images.txt
    data/detection/val_images.txt
    rtdetr-l.pt                          (COCO pretrained weights)

OUTPUT:
    ml/optimization/pso_best.json        (best hyperparameters found)
    ml/optimization/pso_history.json     (full search history for analysis)
    ml/optimization/pso_checkpoint.json  (latest state, for resume)

NEXT STEP:
    python ml/detection/train.py         (automatically loads pso_best.json)
"""

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from ultralytics import RTDETR

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────────────
PRETRAINED   = ROOT / "rtdetr-l.pt"
DATA_YAML    = ROOT / "data" / "detection" / "dataset.yaml"
OPT_DIR      = ROOT / "ml" / "optimization"
PSO_BEST     = OPT_DIR / "pso_best.json"
PSO_HISTORY  = OPT_DIR / "pso_history.json"
PSO_CKPT     = OPT_DIR / "pso_checkpoint.json"
RUNS_DIR     = ROOT / "runs" / "detect" / "pso_trials"

OPT_DIR.mkdir(parents=True, exist_ok=True)

# ── Search space ───────────────────────────────────────────────────────────────
# Each entry: (name, lower_bound, upper_bound, is_log_scale)
# Log scale = True means PSO operates in log space (better for LR, WD)
SEARCH_SPACE = [
    ("lr0",           1e-5,  5e-4,  True ),
    ("weight_decay",  1e-5,  1e-3,  True ),
    ("warmup_epochs", 1.0,   5.0,   False),
    ("mosaic",        0.5,   1.0,   False),
    ("mixup",         0.0,   0.3,   False),
    ("box",           5.0,   10.0,  False),
    ("cls",           0.3,   1.0,   False),
]
N_DIM = len(SEARCH_SPACE)


# ── Encoding helpers ───────────────────────────────────────────────────────────

def encode(params: dict) -> np.ndarray:
    """Convert hyperparameter dict → PSO position vector."""
    pos = np.zeros(N_DIM)
    for i, (name, lb, ub, log) in enumerate(SEARCH_SPACE):
        v = params[name]
        pos[i] = np.log10(v) if log else v
    return pos


def decode(pos: np.ndarray) -> dict:
    """Convert PSO position vector → hyperparameter dict (with clipping)."""
    params = {}
    for i, (name, lb, ub, log) in enumerate(SEARCH_SPACE):
        if log:
            v = 10 ** float(np.clip(pos[i], np.log10(lb), np.log10(ub)))
        else:
            v = float(np.clip(pos[i], lb, ub))
        params[name] = v

    # warmup_epochs must be int
    params["warmup_epochs"] = max(1, round(params["warmup_epochs"]))
    return params


def random_position() -> np.ndarray:
    """Sample a random position uniformly within the (encoded) search space."""
    pos = np.zeros(N_DIM)
    for i, (name, lb, ub, log) in enumerate(SEARCH_SPACE):
        lo = np.log10(lb) if log else lb
        hi = np.log10(ub) if log else ub
        pos[i] = random.uniform(lo, hi)
    return pos


def random_velocity(scale: float = 0.1) -> np.ndarray:
    """Small random initial velocity."""
    vel = np.zeros(N_DIM)
    for i, (name, lb, ub, log) in enumerate(SEARCH_SPACE):
        lo = np.log10(lb) if log else lb
        hi = np.log10(ub) if log else ub
        vel[i] = random.uniform(-(hi - lo) * scale, (hi - lo) * scale)
    return vel


# ── Fitness function ───────────────────────────────────────────────────────────

def evaluate(params: dict, trial_id: str, eval_epochs: int, batch: int) -> float:
    """
    Train RT-DETR-L for `eval_epochs` with `params` and return val mAP50-95.
    Returns 0.0 on error (treat as worst fitness).
    """
    run_name = f"pso_{trial_id}"
    run_dir  = RUNS_DIR / run_name

    try:
        model = RTDETR(str(PRETRAINED))
        results = model.train(
            data          = str(DATA_YAML),
            epochs        = eval_epochs,
            imgsz         = 640,
            batch         = batch,
            workers       = 4,
            optimizer     = "AdamW",
            cos_lr        = True,
            amp           = True,
            cache         = False,
            plots         = False,       # skip plots to save time
            save          = False,       # don't save weights for trials
            val           = True,
            device        = 0,
            verbose       = False,
            project       = str(RUNS_DIR),
            name          = run_name,
            exist_ok      = True,
            freeze        = 23,          # always freeze backbone for short eval
            # hyperparameters from PSO
            lr0           = params["lr0"],
            lrf           = 0.01,
            momentum      = 0.9,
            weight_decay  = params["weight_decay"],
            warmup_epochs = params["warmup_epochs"],
            mosaic        = params["mosaic"],
            mixup         = params["mixup"],
            box           = params["box"],
            cls           = params["cls"],
            dfl           = 1.5,
            label_smoothing = 0.0,
            fliplr        = 0.5,
            hsv_h         = 0.015,
            hsv_s         = 0.7,
            hsv_v         = 0.4,
            degrees       = 5.0,
            translate     = 0.1,
            scale         = 0.5,
        )

        # Extract mAP50-95 from results
        # Ultralytics returns a Results object; metrics are in results.results_dict
        map_val = 0.0
        if hasattr(results, "results_dict"):
            map_val = float(results.results_dict.get("metrics/mAP50-95(B)", 0.0))
        elif hasattr(results, "maps"):
            map_val = float(np.mean(results.maps))

        # Clean up trial run directory to save disk space
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        return map_val

    except Exception as e:
        logger.warning(f"  Trial {trial_id} failed: {e}")
        if run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)
        return 0.0


# ── PSO ───────────────────────────────────────────────────────────────────────

class Particle:
    def __init__(self):
        self.pos    = random_position()
        self.vel    = random_velocity()
        self.pbest  = self.pos.copy()
        self.pbest_fitness = -1.0

    def to_dict(self):
        return {
            "pos":            self.pos.tolist(),
            "vel":            self.vel.tolist(),
            "pbest":          self.pbest.tolist(),
            "pbest_fitness":  self.pbest_fitness,
        }

    @classmethod
    def from_dict(cls, d):
        p = cls.__new__(cls)
        p.pos           = np.array(d["pos"])
        p.vel           = np.array(d["vel"])
        p.pbest         = np.array(d["pbest"])
        p.pbest_fitness = d["pbest_fitness"]
        return p


def run_pso(
    n_particles:  int = 15,
    n_iterations: int = 10,
    eval_epochs:  int = 5,
    resume:       bool = False,
    seed:         int = 42,
):
    random.seed(seed)
    np.random.seed(seed)

    # PSO coefficients
    w_max, w_min = 0.9, 0.4   # inertia weight (decays over iterations)
    c1, c2       = 1.5, 1.5   # cognitive + social coefficients

    # Determine safe batch size
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch = 2 if vram < 4.5 else 4
    else:
        logger.error("No CUDA GPU found.")
        sys.exit(1)

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}, batch={batch}")

    history = []   # list of {iteration, particle, params, fitness}
    start_iter = 0

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume and PSO_CKPT.exists():
        ckpt = json.loads(PSO_CKPT.read_text())
        particles = [Particle.from_dict(d) for d in ckpt["particles"]]
        gbest_pos     = np.array(ckpt["gbest_pos"])
        gbest_fitness = ckpt["gbest_fitness"]
        start_iter    = ckpt["iteration"] + 1
        history       = ckpt.get("history", [])
        logger.success(f"Resumed from iteration {start_iter}, best mAP={gbest_fitness:.4f}")
    else:
        particles     = [Particle() for _ in range(n_particles)]
        gbest_pos     = particles[0].pos.copy()
        gbest_fitness = -1.0

    logger.info("── PSO Hyperparameter Search ──────────────────────────────")
    logger.info(f"  Particles  : {n_particles}")
    logger.info(f"  Iterations : {n_iterations}")
    logger.info(f"  Eval epochs: {eval_epochs} per trial")
    logger.info(f"  Dimensions : {N_DIM}")
    logger.info(f"  Total evals: {n_particles * n_iterations}")
    logger.info(f"  Search space:")
    for name, lb, ub, log in SEARCH_SPACE:
        scale = "(log)" if log else "     "
        logger.info(f"    {name:20s} [{lb}, {ub}] {scale}")

    # ── Main PSO loop ─────────────────────────────────────────────────────────
    for iteration in range(start_iter, n_iterations):
        w = w_max - (w_max - w_min) * iteration / max(n_iterations - 1, 1)
        logger.info(f"\n── Iteration {iteration + 1}/{n_iterations}  (w={w:.3f}) ─────────")

        for p_idx, particle in enumerate(particles):
            params = decode(particle.pos)
            trial_id = f"i{iteration:02d}_p{p_idx:02d}"

            logger.info(f"  Particle {p_idx+1:2d}/{n_particles}  trial={trial_id}")
            logger.info(f"    lr0={params['lr0']:.2e}  wd={params['weight_decay']:.2e}  "
                        f"warmup={params['warmup_epochs']}  "
                        f"mosaic={params['mosaic']:.2f}  mixup={params['mixup']:.2f}  "
                        f"box={params['box']:.1f}  cls={params['cls']:.2f}")

            t0 = time.time()
            fitness = evaluate(params, trial_id, eval_epochs, batch)
            elapsed = time.time() - t0

            logger.info(f"    mAP50-95 = {fitness:.4f}  ({elapsed/60:.1f} min)")

            # Update personal best
            if fitness > particle.pbest_fitness:
                particle.pbest         = particle.pos.copy()
                particle.pbest_fitness = fitness
                logger.info(f"    ↑ New personal best: {fitness:.4f}")

            # Update global best
            if fitness > gbest_fitness:
                gbest_fitness = fitness
                gbest_pos     = particle.pos.copy()
                logger.success(f"    ★ New GLOBAL best: {fitness:.4f}")
                # Save best immediately
                best_params = decode(gbest_pos)
                PSO_BEST.write_text(json.dumps(best_params, indent=2))
                logger.success(f"    Saved → {PSO_BEST.name}")

            history.append({
                "iteration": iteration,
                "particle":  p_idx,
                "params":    params,
                "fitness":   fitness,
                "elapsed_s": elapsed,
            })

        # ── Update velocities and positions ───────────────────────────────────
        for particle in particles:
            r1 = np.random.rand(N_DIM)
            r2 = np.random.rand(N_DIM)
            particle.vel = (
                w * particle.vel
                + c1 * r1 * (particle.pbest - particle.pos)
                + c2 * r2 * (gbest_pos     - particle.pos)
            )
            particle.pos = particle.pos + particle.vel

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt = {
            "iteration":      iteration,
            "gbest_pos":      gbest_pos.tolist(),
            "gbest_fitness":  gbest_fitness,
            "particles":      [p.to_dict() for p in particles],
            "history":        history,
        }
        PSO_CKPT.write_text(json.dumps(ckpt, indent=2))

        logger.info(f"\n  Best so far: mAP50-95 = {gbest_fitness:.4f}")
        logger.info(f"  Best params: {decode(gbest_pos)}")

    # ── Final output ──────────────────────────────────────────────────────────
    best_params = decode(gbest_pos)

    PSO_BEST.write_text(json.dumps(best_params, indent=2))
    PSO_HISTORY.write_text(json.dumps(history, indent=2))

    logger.success("\n── PSO Complete ───────────────────────────────────────────")
    logger.success(f"  Best mAP50-95 : {gbest_fitness:.4f}")
    logger.success(f"  Best params   :")
    for k, v in best_params.items():
        logger.success(f"    {k:20s} = {v}")
    logger.success(f"\n  Saved → {PSO_BEST}")
    logger.success(f"  History → {PSO_HISTORY}")
    logger.success(f"\n  Next step: python ml/detection/train.py")

    return best_params


# ── Entry point ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="PSO hyperparameter search for RT-DETR-L"
    )
    p.add_argument("--particles",   type=int,  default=15,
                   help="Number of particles (default: 15)")
    p.add_argument("--iterations",  type=int,  default=10,
                   help="Number of PSO iterations (default: 10)")
    p.add_argument("--eval_epochs", type=int,  default=5,
                   help="Training epochs per fitness evaluation (default: 5)")
    p.add_argument("--resume",      action="store_true",
                   help="Resume from pso_checkpoint.json")
    p.add_argument("--seed",        type=int,  default=42,
                   help="Random seed (default: 42)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pso(
        n_particles  = args.particles,
        n_iterations = args.iterations,
        eval_epochs  = args.eval_epochs,
        resume       = args.resume,
        seed         = args.seed,
    )