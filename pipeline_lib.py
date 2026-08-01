"""Shared definitions for the churn pipeline.

The analysis lives in `notebooks/churn_analysis.ipynb`. This module exists for
one reason: a pickled sklearn Pipeline stores its custom steps *by import
path*, so any class inside the saved model has to be importable at load time.
If `FeatureEngineer` were defined in the notebook it would pickle as
`__main__.FeatureEngineer` and the API could never load it.

So the two custom transformers and the column lists they depend on live here,
and everything else - loading, EDA, splitting, model search, evaluation,
explainability - happens in the notebook.

Nothing in here should need to change to re-run the analysis.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------------------------------------------------------
# Paths and constants shared by the notebook and the serving layer
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"

PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "validation.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"

FINAL_PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42
TARGET = "Churn"
ID_COLUMN = "customerID"

BINARY_FEATURES = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
MULTICLASS_FEATURES = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies",
]

ENGINEERED_NUMERIC = ["avg_monthly_charge", "service_count", "charge_tenure_ratio"]
ENGINEERED_FLAGS = ["has_security_backup", "high_value_short_tenure"]
ENGINEERED_CATEGORICAL = ["tenure_group"]
FLAG_FEATURES = ["SeniorCitizen"] + ENGINEERED_FLAGS


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Force the numeric base columns to numeric dtype.

    Inference data does not pass through the notebook's cleaning step, so a raw
    CSV upload can still deliver TotalCharges as strings with blanks for
    tenure-0 customers. Same semantics as training: blanks -> NaN, and
    TotalCharges NaN -> 0 (brand-new customers who have not been billed).
    """
    df = df.copy()
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def engineer_features(df: pd.DataFrame, charge_threshold: Optional[float] = None) -> pd.DataFrame:
    """Add the six engineered features.

    `charge_threshold` must come from training data when this runs in a
    pipeline - see FeatureEngineer. Passing None falls back to the median of
    `df` itself, which is only meaningful for whole-dataset EDA.
    """
    df = df.copy()

    # Tenure buckets: churn risk changes in phases, not linearly. The last bin
    # is open-ended so tenure above 72 months never produces NaN.
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 36, 48, 60, np.inf],
        labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61+"],
        include_lowest=True,
    ).astype(str)

    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["charge_tenure_ratio"] = df["MonthlyCharges"] / df["tenure"].clip(lower=1)

    # A service counts as active unless it reads No / No phone service /
    # No internet service. Checking for "Yes" instead would silently skip
    # InternetService, whose values are DSL / Fiber optic / No.
    available = [c for c in SERVICE_COLUMNS if c in df.columns]
    if available:
        normalized = df[available].astype(str).apply(lambda c: c.str.strip().str.lower())
        inactive = {"no", "no phone service", "no internet service", "nan", ""}
        df["service_count"] = (~normalized.isin(inactive)).sum(axis=1).astype(int)
    else:
        df["service_count"] = 0

    def _is_yes(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].astype(str).str.strip().str.lower() == "yes"
        # Index-aligned False, unlike df.get(col, pd.Series()) which yields NaN
        return pd.Series(False, index=df.index)

    df["has_security_backup"] = (_is_yes("OnlineSecurity") & _is_yes("OnlineBackup")).astype(int)

    if charge_threshold is None:
        charge_threshold = df["MonthlyCharges"].median()
    df["high_value_short_tenure"] = (
        (df["MonthlyCharges"] > charge_threshold) & (df["tenure"] < 12)
    ).astype(int)

    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """First step of the model pipeline.

    `fit` learns the MonthlyCharges median from the training fold only, so
    cross-validation stays leakage-free and a single-row API request gets the
    identical feature definition it would have had during training.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        if "MonthlyCharges" not in X.columns:
            raise ValueError("FeatureEngineer requires a 'MonthlyCharges' column.")
        X = coerce_numeric_columns(X)
        self.monthly_charges_median_ = float(X["MonthlyCharges"].median())
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = coerce_numeric_columns(X)
        return engineer_features(X, charge_threshold=self.monthly_charges_median_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            input_features = self.feature_names_in_
        # Must match the order transform() appends them in, or downstream
        # feature labels silently point at the wrong column.
        engineered = [
            "tenure_group", "avg_monthly_charge", "charge_tenure_ratio",
            "service_count", "has_security_backup", "high_value_short_tenure",
        ]
        new = [f for f in engineered if f not in list(input_features)]
        return np.concatenate(
            [np.asarray(input_features, dtype=object), np.asarray(new, dtype=object)]
        )


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Map Yes/No and Male/Female to 1/0, one column in, one column out.

    Unmapped values raise rather than pass through: failing loudly beats
    feeding garbage into a model that is about to make a retention decision.
    """

    _MAPPING = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}

    def fit(self, X: pd.DataFrame, y=None) -> "BinaryEncoder":
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == object:
                stripped = X[col].astype(str).str.strip()
                mapped = stripped.map(self._MAPPING)
                unknown = mapped.isna() & X[col].notna()
                if unknown.any():
                    bad = sorted(stripped[unknown].unique().tolist())
                    raise ValueError(
                        f"BinaryEncoder: column '{col}' contains values that are not "
                        f"binary Yes/No or Male/Female: {bad}"
                    )
                X[col] = mapped.fillna(0).astype(int)
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_
