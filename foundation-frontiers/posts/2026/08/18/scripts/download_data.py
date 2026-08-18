"""
Download the Home Credit dataset from Kaggle.

Requirements:
  pip install kaggle

Authentication (pick one):
  1. Set env var:  export KAGGLE_API_TOKEN="KGAT_..."
  2. Or place kaggle.json in ~/.kaggle/kaggle.json

Usage:
  python scripts/download_data.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

COMPETITION: str = "home-credit-credit-risk-model-stability"
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
DEST: Path = DATA_DIR / COMPETITION


def load_env_token() -> None:
    """Load KAGGLE_API_TOKEN from .env if not already set."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> None:
    load_env_token()

    if (
        not os.environ.get("KAGGLE_API_TOKEN")
        and not Path("~/.kaggle/kaggle.json").expanduser().exists()
    ):
        print("Error: No Kaggle credentials found.")
        print("Set KAGGLE_API_TOKEN env var or place kaggle.json in ~/.kaggle/")
        sys.exit(1)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path: Path = DATA_DIR / f"{COMPETITION}.zip"

    if DEST.exists() and any(DEST.rglob("*.parquet")):
        print(f"Dataset already exists at {DEST}")
        print("Delete the directory to re-download.")
        return

    print(f"Downloading {COMPETITION}...")
    subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(DATA_DIR)],
        check=True,
    )

    print(f"Extracting to {DEST}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DEST)

    zip_path.unlink()
    print(f"Done. Dataset at {DEST}")


if __name__ == "__main__":
    main()
