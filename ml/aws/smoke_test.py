"""
ml/aws/smoke_test.py
--------------------
Prove the GPU stack actually works before committing hours of training to it.

WHY

    `pip install` on a managed SageMaker image routinely upgrades numpy past what
    the preinstalled packages pin, and prints a warning that is easy to scroll past:

        numba requires numpy<2.5, but you have numpy 2.5.1
        sagemaker-studio requires numpy<2.3.0, but you have numpy 2.5.1

    Most of the time this is harmless. Occasionally it is not: torch, OpenCV and
    Ultralytics all cross the C extension boundary with numpy, and an ABI mismatch
    surfaces as a segfault or a bizarre dtype error partway through the first epoch -
    which on a 20-hour account is an expensive way to learn about it.

    Twenty seconds here is worth it. Every check below exercises the exact interop
    path training uses, not just `import x`.

USAGE
    python ml/aws/smoke_test.py
    python ml/aws/smoke_test.py --quick    # skip the tiny training step
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

PASS, FAIL = "  ok  ", " FAIL "
results: list[tuple[bool, str]] = []


def check(name: str, fn) -> bool:
    try:
        detail = fn()
        results.append((True, f"{name}: {detail}"))
        print(f"{PASS}{name}: {detail}")
        return True
    except Exception as exc:
        results.append((False, f"{name}: {exc}"))
        print(f"{FAIL}{name}: {type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
def t_versions() -> str:
    import numpy as np
    import torch
    return f"torch {torch.__version__}, numpy {np.__version__}"


def t_cuda() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Training would silently run ~50x slower on CPU."
        )
    p = torch.cuda.get_device_properties(0)
    return f"{p.name}, {p.total_memory / 1024**3:.0f} GB"


def t_torch_numpy() -> str:
    """
    The actual ABI surface. A numpy too new for the torch build fails HERE, not at
    import, which is why importing both successfully proves nothing.
    """
    import numpy as np
    import torch

    a = np.random.rand(8, 3, 32, 32).astype("float32")
    t = torch.from_numpy(a)                 # numpy -> torch, zero copy
    back = t.numpy()                        # torch -> numpy
    if not np.allclose(a, back):
        raise RuntimeError("round-trip changed the data")
    g = t.cuda()                            # host -> device
    d = g.cpu().numpy()                     # device -> host -> numpy
    if not np.allclose(a, d):
        raise RuntimeError("GPU round-trip changed the data")
    return f"round-trip clean, {a.nbytes / 1024:.0f} KB"


def t_cuda_math() -> str:
    """A real fp16 matmul: catches a broken CUDA/cuDNN install that imports fine."""
    import torch

    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    y = (x @ x).float()
    if not torch.isfinite(y).all():
        raise RuntimeError("fp16 matmul produced non-finite values")
    torch.cuda.synchronize()
    return "fp16 matmul ok (this is the path AMP training uses)"


def t_cv2() -> str:
    """OpenCV is a C extension linked against numpy; the pipeline uses it everywhere."""
    import cv2
    import numpy as np

    img = (np.random.rand(64, 64, 3) * 255).astype("uint8")
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if th.shape != (64, 64):
        raise RuntimeError("unexpected shape from OpenCV")
    return f"{cv2.__version__}, colour convert + Otsu ok"


def t_pillow() -> str:
    from PIL import Image
    import numpy as np

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.jpg"
        Image.new("RGB", (64, 48)).save(p)
        with Image.open(p) as im:
            if im.size != (64, 48):
                raise RuntimeError("size mismatch")
            arr = np.asarray(im)
    return f"encode/decode ok, array {arr.shape}"


def t_ultralytics_import() -> str:
    import ultralytics
    from ultralytics import RTDETR, YOLO  # noqa: F401
    return f"{ultralytics.__version__}, RTDETR and YOLO importable"


def t_harness() -> str:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ml.research.class_sets import CLASS_SETS
    from ml.research.experiments import REGISTRY, check_registry

    problems = check_registry()
    if problems:
        raise RuntimeError(f"registry problems: {problems[:2]}")
    return f"{len(REGISTRY)} experiments, {len(CLASS_SETS)} class sets, registry clean"


def t_tiny_train() -> str:
    """
    One real forward+backward on the GPU through Ultralytics.

    This is the only check that exercises the full stack the way training does:
    dataloader, augmentation, autocast, optimiser step. If numpy or OpenCV are going
    to break a run, they break here rather than at epoch 3.
    """
    import numpy as np
    import torch
    from PIL import Image
    from ultralytics import YOLO

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for split in ("train", "val"):
            (root / split / "images").mkdir(parents=True)
            (root / split / "labels").mkdir(parents=True)
            for i in range(4):
                arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
                Image.fromarray(arr).save(root / split / "images" / f"{i}.jpg")
                (root / split / "labels" / f"{i}.txt").write_text("0 0.5 0.5 0.3 0.3\n")
        (root / "d.yaml").write_text(
            f"path: {root}\ntrain: train/images\nval: val/images\nnc: 1\nnames:\n  0: x\n"
        )
        m = YOLO("yolo11n.pt")
        m.train(data=str(root / "d.yaml"), epochs=1, imgsz=64, batch=2,
                device=0, workers=0, verbose=False, plots=False, val=False,
                project=str(root / "r"), name="smoke", exist_ok=True)
    torch.cuda.synchronize()
    return "one epoch through the real training loop on GPU"


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the GPU stack before training")
    ap.add_argument("--quick", action="store_true",
                    help="skip the tiny training run (downloads a small checkpoint)")
    args = ap.parse_args()

    print("=" * 66)
    print("GPU stack smoke test")
    print("=" * 66)

    checks = [
        ("versions", t_versions),
        ("cuda available", t_cuda),
        ("torch <-> numpy", t_torch_numpy),
        ("cuda fp16 math", t_cuda_math),
        ("opencv", t_cv2),
        ("pillow", t_pillow),
        ("ultralytics", t_ultralytics_import),
        ("research harness", t_harness),
    ]
    if not args.quick:
        checks.append(("tiny GPU training", t_tiny_train))

    for name, fn in checks:
        check(name, fn)

    failed = [m for ok, m in results if not ok]
    print("=" * 66)
    if not failed:
        print(f"all {len(results)} checks passed - the stack is safe to train on")
        return 0

    print(f"{len(failed)} of {len(results)} checks FAILED:")
    for m in failed:
        print("  - " + m)
    print("""
Most likely cause: pip upgraded numpy past what the preinstalled torch/OpenCV
build supports. Pin it back, then re-run this:

    pip install "numpy<2.3" --quiet
    python ml/aws/smoke_test.py

If CUDA specifically is unavailable:

    pip install --force-reinstall torch torchvision \\
        --index-url https://download.pytorch.org/whl/cu121
""")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
