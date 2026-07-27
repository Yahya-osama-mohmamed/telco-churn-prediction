"""
Data Loader — Download and load the Telco Customer Churn dataset.

Handles:
- Downloading from a direct URL (no Kaggle credentials required)
- Loading and initial validation (schema checks, dtype verification)
- Basic data profiling (shape, missing values, duplicates)

The IBM Telco Customer Churn dataset contains 7,043 customer records
with 21 columns covering demographics, account info, and services.
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Tuple, Dict, Any

from src.config import DATASET_URL, RAW_DATA_FILE, TARGET, ID_COLUMN
from src.logger import get_logger

logger = get_logger(__name__)


# Expected columns in the raw dataset — used as a sanity check
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def download_dataset(url: str = DATASET_URL, dest: Path = RAW_DATA_FILE) -> Path:
    """
    Download the dataset from a direct URL if not already present.

    Args:
        url: Direct download URL for the CSV file.
        dest: Local file path to save the downloaded CSV.

    Returns:
        Path to the downloaded (or existing) file.

    Raises:
        requests.HTTPError: If the download fails.
    """
    if dest.exists():
        logger.info(f"Dataset already exists at {dest} — skipping download.")
        return dest

    logger.info(f"Downloading dataset from {url}...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    dest.write_bytes(response.content)
    logger.info(f"Dataset saved to {dest} ({dest.stat().st_size / 1024:.1f} KB)")
    return dest


def load_raw_data(filepath: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load the raw CSV dataset and perform initial type fixes.

    Known issue: 'TotalCharges' column contains blank strings (' ')
    for customers with tenure=0. These are converted to NaN here.

    Args:
        filepath: Path to the raw CSV file.

    Returns:
        DataFrame with basic type corrections applied.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the schema doesn't match expectations.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Run download_dataset() first."
        )

    logger.info(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    # --- Schema validation ---
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    # --- Fix TotalCharges: blank strings → NaN → float ---
    # 11 rows have ' ' (space) in TotalCharges — these are new customers
    # with tenure=0 who haven't been billed yet
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    logger.info(
        f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns"
    )
    return df


def get_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive data profile for EDA.

    Returns a dictionary with:
    - Shape, dtypes, memory usage
    - Missing value counts and percentages
    - Duplicate row count
    - Class distribution of the target variable
    - Basic statistics

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary containing profile metrics.
    """
    n_rows, n_cols = df.shape

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    missing_info = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }

    # Duplicates (by customerID)
    n_duplicates = df.duplicated(subset=[ID_COLUMN]).sum() if ID_COLUMN in df.columns else 0

    # Class distribution
    if TARGET in df.columns:
        class_dist = df[TARGET].value_counts().to_dict()
        class_pct = df[TARGET].value_counts(normalize=True).mul(100).round(2).to_dict()
    else:
        class_dist = {}
        class_pct = {}

    profile = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": missing_info,
        "n_missing_total": int(missing.sum()),
        "n_duplicates": int(n_duplicates),
        "class_distribution": class_dist,
        "class_percentage": class_pct,
    }

    logger.info(
        f"Profile: {n_rows} rows, {n_cols} cols, "
        f"{len(missing_info)} cols with missing values, "
        f"{n_duplicates} duplicate IDs"
    )

    return profile


def load_and_validate() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Complete data loading pipeline: download → load → profile.

    Returns:
        Tuple of (DataFrame, profile dictionary).
    """
    download_dataset()
    df = load_raw_data()
    profile = get_data_profile(df)
    return df, profile
