"""
scripts/download_datasets.py

Downloads all public datasets used for training.
Each dataset has its own function so you can run them independently.

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --rdd2022
    python scripts/download_datasets.py --gaps
    python scripts/download_datasets.py --pothole600
    python scripts/download_datasets.py --cfd

RDD2022 / CRDDC2022 require you to accept their terms on the official
repository first. The download URLs are provided as constants below —
replace them if the hosting location changes.

All datasets are saved to data/datasets/<name>/
"""

import os
import sys
import argparse
import zipfile
import tarfile
import shutil
import requests
from tqdm import tqdm
from loguru import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Dataset URLs ─────────────────────────────────────────────────────────────
# These are the canonical sources as of 2024.
# If a URL breaks, check the dataset's GitHub repo or paper for the latest link.

DATASETS = {
    "rdd2022": {
        "description": "Road Damage Detection 2022 — multi-national, 47k images, 4 damage classes",
        "paper": "https://arxiv.org/abs/2209.08538",
        "source": "https://github.com/sekilab/RoadDamageDetector",
        # Direct download from Zenodo (official hosting):
        "url": "https://zenodo.org/record/7547905/files/RDD2022.zip",
        "filename": "RDD2022.zip",
        "extract_to": "data/datasets/rdd2022",
        "instructions": (
            "If the direct URL fails, download manually from:\n"
            "  https://zenodo.org/record/7547905\n"
            "Accept the terms, download RDD2022.zip, place it in data/datasets/rdd2022/"
        ),
    },
    "gaps": {
        "description": "GAPs — German Asphalt Pavement distress dataset, ~2k high-res images",
        "paper": "https://arxiv.org/abs/1903.05445",
        "source": "https://www.tu-ilmenau.de/neurob/data-sets-code/gaps/",
        "url": "https://www.tu-ilmenau.de/fileadmin/Bereiche/IA/neurob/Datasets/GAPs/GAPs_v2.zip",
        "filename": "GAPs_v2.zip",
        "extract_to": "data/datasets/gaps",
        "instructions": (
            "If the direct URL fails, register and download from:\n"
            "  https://www.tu-ilmenau.de/neurob/data-sets-code/gaps/\n"
            "Place the zip in data/datasets/gaps/"
        ),
    },
    "pothole600": {
        "description": "Pothole-600 — 600 images with segmentation masks (COCO polygon format)",
        "paper": "https://arxiv.org/abs/2107.05844",
        "source": "https://github.com/JackTruskowski/pothole-detection",
        # Hosted on Kaggle — use kaggle API or download manually
        "url": None,
        "filename": None,
        "extract_to": "data/datasets/pothole600",
        "instructions": (
            "Pothole-600 must be downloaded manually:\n"
            "  1. Go to https://www.kaggle.com/datasets/andrewmvd/pothole-detection\n"
            "  2. Download the dataset zip\n"
            "  3. Extract to data/datasets/pothole600/\n"
            "     Expected structure:\n"
            "     data/datasets/pothole600/images/\n"
            "     data/datasets/pothole600/annotations/"
        ),
    },
    "cfd": {
        "description": "Crack Forest Dataset — 118 images, pixel-level crack segmentation",
        "paper": "https://arxiv.org/abs/1509.00116",
        "source": "https://github.com/cuilimeng/CrackForest-dataset",
        "url": "https://github.com/cuilimeng/CrackForest-dataset/archive/refs/heads/master.zip",
        "filename": "CrackForest-dataset.zip",
        "extract_to": "data/datasets/cfd",
        "instructions": (
            "CFD can also be cloned directly:\n"
            "  git clone https://github.com/cuilimeng/CrackForest-dataset data/datasets/cfd"
        ),
    },
}

# ── Download helper ───────────────────────────────────────────────────────────

def download_file(url: str, dest_path: str) -> bool:
    """
    Stream-download a file with a progress bar.
    Returns True on success, False on failure.
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        with open(dest_path, "wb") as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)

        return True

    except requests.exceptions.RequestException as e:
        logger.error(f"Download failed: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str):
    """Extract .zip or .tar.gz archive."""
    os.makedirs(extract_to, exist_ok=True)
    logger.info(f"Extracting {archive_path} → {extract_to}")

    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as z:
            z.extractall(extract_to)
    elif archive_path.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive_path, "r:gz") as t:
            t.extractall(extract_to)
    else:
        logger.warning(f"Unknown archive format: {archive_path}. Extract manually.")
        return

    logger.info("Extraction complete.")


def download_dataset(name: str):
    info = DATASETS[name]
    logger.info(f"\n{'─' * 60}")
    logger.info(f"Dataset: {name}")
    logger.info(f"Description: {info['description']}")
    logger.info(f"Paper: {info.get('paper', 'N/A')}")

    extract_to = info["extract_to"]
    os.makedirs(extract_to, exist_ok=True)

    if info["url"] is None:
        # Manual download required
        logger.warning(f"Manual download required for '{name}':")
        logger.warning(info["instructions"])
        return

    dest_path = os.path.join(extract_to, info["filename"])

    # Skip if already downloaded
    if os.path.exists(dest_path):
        logger.info(f"Archive already exists: {dest_path}. Skipping download.")
    else:
        logger.info(f"Downloading from: {info['url']}")
        success = download_file(info["url"], dest_path)
        if not success:
            logger.warning(f"Automatic download failed. Manual instructions:")
            logger.warning(info["instructions"])
            return

    extract_archive(dest_path, extract_to)
    logger.success(f"'{name}' ready at {extract_to}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download training datasets for the Cluj Urban Monitor."
    )
    parser.add_argument("--all",        action="store_true", help="Download all datasets")
    parser.add_argument("--rdd2022",    action="store_true", help="Download RDD2022")
    parser.add_argument("--gaps",       action="store_true", help="Download GAPs")
    parser.add_argument("--pothole600", action="store_true", help="Show Pothole-600 instructions")
    parser.add_argument("--cfd",        action="store_true", help="Download CrackForest dataset")

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        sys.exit(0)

    targets = []
    if args.all:
        targets = list(DATASETS.keys())
    else:
        if args.rdd2022:    targets.append("rdd2022")
        if args.gaps:       targets.append("gaps")
        if args.pothole600: targets.append("pothole600")
        if args.cfd:        targets.append("cfd")

    for name in targets:
        download_dataset(name)

    logger.success("\nAll requested datasets processed.")
    logger.info("Next step: run python ml/detection/data_prep.py to remap labels.")


if __name__ == "__main__":
    main()