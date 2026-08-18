"""
Prepare and upload the processed Home Credit dataset to HuggingFace Hub.

Reads raw parquet files, merges features + demographics, and saves a single
processed parquet. Optionally uploads to HF Hub.

Prerequisites:
  brew install hf
  hf auth login

Usage:
  python scripts/prepare_dataset.py                  # prepare only
  python scripts/prepare_dataset.py --upload          # prepare + upload
  python scripts/prepare_dataset.py --repo deburky/home-credit-credit-risk-model-stability
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# -------------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA: Path = PROJECT_ROOT / "data" / "home-credit-credit-risk-model-stability"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE: Path = PROCESSED_DIR / "home_credit_processed.parquet"

# -------------------------------------------------------------------------------
# Feature definitions
# -------------------------------------------------------------------------------

FEATURES_DPD: list[str] = [
    "maxdpdfrom6mto36m_3546853P",
    "maxdpdlast12m_727P",
    "maxdpdlast24m_143P",
    "maxdpdlast3m_392P",
    "maxdpdlast6m_474P",
]

FEATURES_USER_PROFILE: list[str] = [
    "numinstls_657L",
    "numinstlsallpaid_934L",
    "pctinstlsallpaidlat10d_839L",
    "totalsettled_863A",
    "totaldebt_9A",
    "currdebt_22A",
    "credamount_770A",
    "mobilephncnt_593L",
    "homephncnt_628L",
    "numactivecreds_622L",
    "applicationcnt_361L",
    "applications30d_658L",
    "applicationscnt_1086L",
    "applicationscnt_464L",
    "applicationscnt_629L",
    "avgdbddpdlast24m_3658932P",
    "amtinstpaidbefduel24m_4187115A",
    "maxdbddpdlast1m_3658939P",
    "maxdbddpdtollast12m_3658940P",
    "maxdbddpdtollast6m_4187119P",
    "numinstpaidlate1d_3546852L",
]

FEATURES_CB: list[str] = [
    "numberofqueries_373L",
    "days120_123L",
    "days180_256L",
    "days30_165L",
    "days90_310L",
    "days360_512L",
]

ALL_FEATURES: list[str] = FEATURES_DPD + FEATURES_CB + FEATURES_USER_PROFILE


def prepare() -> Path:
    """Load raw parquets, merge, and save processed dataset."""
    if not RAW_DATA.exists():
        print(f"Raw data not found at {RAW_DATA}")
        print("Run: python scripts/download_data.py")
        sys.exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------------
    # Load and merge feature tables
    # -------------------------------------------------------------------------------
    print("Loading raw parquet files...")
    train_dir: str = str(RAW_DATA / "parquet_files" / "train")

    labels: pd.DataFrame = pd.read_parquet(f"{train_dir}/train_base.parquet")
    dpd: pd.DataFrame = pd.read_parquet(
        f"{train_dir}/train_static_0_1.parquet",
        columns=FEATURES_DPD + FEATURES_USER_PROFILE + ["case_id"],
    )
    cb: pd.DataFrame = pd.read_parquet(
        f"{train_dir}/train_static_cb_0.parquet",
        columns=FEATURES_CB + ["case_id"],
    )

    df: pd.DataFrame = labels.merge(dpd, on="case_id").merge(cb, on="case_id")

    # -------------------------------------------------------------------------------
    # Protected attributes from person table
    # -------------------------------------------------------------------------------
    # Demographic and categorical attributes from person_1
    # -------------------------------------------------------------------------------
    print("Adding demographic attributes...")
    person: pd.DataFrame = pd.read_parquet(
        f"{train_dir}/train_person_1.parquet",
        columns=[
            "case_id", "num_group1",
            "sex_738L", "birth_259D",
            "education_927M", "incometype_1044T",
            "familystate_447L", "empl_employedtotal_800L",
            "language1_981M", "mainoccupationinc_384A",
        ],
    )
    person = person[person["num_group1"] == 0].drop(columns=["num_group1"])

    person["birth_259D"] = pd.to_datetime(person["birth_259D"], errors="coerce")
    reference_date: pd.Timestamp = pd.Timestamp("2024-01-01")
    person["age"] = ((reference_date - person["birth_259D"]).dt.days / 365.25).round(0)
    person = person.drop(columns=["birth_259D"])

    df = df.merge(person, on="case_id", how="left")

    # -------------------------------------------------------------------------------
    # Credit bureau categoricals from static_cb_0
    # -------------------------------------------------------------------------------
    print("Adding credit bureau attributes...")
    cb_cat: pd.DataFrame = pd.read_parquet(
        f"{train_dir}/train_static_cb_0.parquet",
        columns=[
            "case_id",
            "maritalst_385M",         # marital status (CB, masked, 6 levels)
            "requesttype_4525192L",   # DEDUCTION / PENSION / SOCIAL (45% coverage)
            "description_5085714M",   # product description (2 levels)
        ],
    )
    df = df.merge(cb_cat, on="case_id", how="left")

    # -------------------------------------------------------------------------------
    # Convert date columns
    # -------------------------------------------------------------------------------
    df["date_decision"] = pd.to_datetime(df["date_decision"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["MONTH"] = pd.to_datetime(df["MONTH"].astype(str), format="%Y%m").dt.strftime("%Y-%m")

    # -------------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------------
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved processed dataset: {OUTPUT_FILE}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Size:  {OUTPUT_FILE.stat().st_size / 1e6:.1f} MB")
    return OUTPUT_FILE


def upload(repo: str) -> None:
    """Upload processed directory to HuggingFace Hub."""
    if not OUTPUT_FILE.exists():
        print("No processed file found. Run prepare() first.")
        sys.exit(1)

    load_dotenv(PROJECT_ROOT / ".env")
    hf_token: str | None = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: Set HF_WRITE_TOKEN in .env or HF_TOKEN env var")
        sys.exit(1)

    env: dict[str, str] = {**os.environ, "HF_TOKEN": hf_token}

    print(f"Uploading to {repo}...")
    subprocess.run(
        ["hf", "upload", repo, str(PROCESSED_DIR), "--repo-type=dataset"],
        check=True,
        env=env,
    )
    print(f"Done. Dataset at https://huggingface.co/datasets/{repo}")


def main() -> None:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Prepare and optionally upload Home Credit dataset"
    )
    parser.add_argument(
        "--upload", action="store_true", help="Upload to HuggingFace Hub after preparing"
    )
    parser.add_argument(
        "--repo",
        default="deburky/home-credit-credit-risk-model-stability",
        help="HuggingFace dataset repo (default: deburky/home-credit-credit-risk-model-stability)",
    )
    args: argparse.Namespace = parser.parse_args()

    prepare()

    if args.upload:
        upload(args.repo)


if __name__ == "__main__":
    main()
