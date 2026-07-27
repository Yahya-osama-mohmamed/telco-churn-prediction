"""
Feature Engineering — Create business-meaningful features from raw data.

New features are designed to capture domain knowledge about telecom churn:
- Customer lifecycle stage (tenure grouping)
- Service engagement level (service count)
- Spending patterns (average monthly charge)
- Risk indicators (high-value short-tenure, security status)

The `FeatureEngineer` transformer is the production entry point: it lives
INSIDE the sklearn Pipeline, so any statistic it needs (the MonthlyCharges
median used by `high_value_short_tenure`) is learned in `fit` on training
data only and re-applied identically at inference time. This prevents both
data leakage during cross-validation and train/serve skew in the API.

The module-level `create_*` functions and `engineer_features` remain
available for EDA and notebooks.
"""

import pandas as pd
import numpy as np
from typing import Optional

from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    SERVICE_COLUMNS, ENGINEERED_NUMERIC, ENGINEERED_FLAGS, ENGINEERED_CATEGORICAL,
)
from src.logger import get_logger

logger = get_logger(__name__)


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce the numeric base columns to numeric dtype.

    Inference data does not pass through DataCleaner (it is not part of the
    serving pipeline), so a raw CSV upload can deliver TotalCharges as
    strings with blanks for tenure-0 customers. This mirrors the cleaning
    semantics: blanks/garbage → NaN, TotalCharges NaN → 0 (new customers);
    remaining NaN is handled by the pipeline's SimpleImputer.
    """
    df = df.copy()
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        if col in df.columns and df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def create_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin tenure into lifecycle groups.

    Rationale: Customer behavior changes non-linearly with tenure.
    New customers (0-12 months) have the highest churn risk,
    while long-term customers (49+ months) are much more loyal.
    Binning captures these phase transitions.

    Groups:
    - 0-12:  New customer (highest churn risk)
    - 13-24: Early adopter
    - 25-36: Established
    - 37-48: Loyal
    - 49-60: Long-term
    - 61+:   Veteran (lowest churn risk)

    The last bin is open-ended so tenure values above 72 months
    (possible in future data) never produce NaN.
    """
    df = df.copy()
    bins = [0, 12, 24, 36, 48, 60, np.inf]
    labels = ["0-12", "13-24", "25-36", "37-48", "49-60", "61+"]
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=bins, labels=labels, include_lowest=True
    ).astype(str)
    return df


def create_avg_monthly_charge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average monthly charge from total charges and tenure.

    Formula: TotalCharges / max(tenure, 1)

    Rationale: This measures spending consistency. A customer paying
    $70/month with a $70 average is stable, while one paying $70/month
    with a $50 average may have recently upgraded (potential satisfaction risk).

    The max(tenure, 1) prevents division by zero for brand-new customers.
    """
    df = df.copy()
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count the number of active services per customer.

    Rationale: Customers with more services have higher switching costs
    (they'd need to find multiple replacements). This is a strong
    predictor of retention — bundled customers are less likely to churn.

    Services counted: PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
    StreamingTV, StreamingMovies.

    A service is active unless its value is 'No' / 'No phone service' /
    'No internet service' (or missing). This rule is what lets
    InternetService count: its values are 'DSL' / 'Fiber optic' / 'No',
    never 'Yes', so a plain equals-'Yes' check would silently ignore the
    internet subscription for every internet customer.
    """
    df = df.copy()
    available_services = [col for col in SERVICE_COLUMNS if col in df.columns]

    if available_services:
        normalized = df[available_services].astype(str).apply(
            lambda col: col.str.strip().str.lower()
        )
        inactive = {"no", "no phone service", "no internet service", "nan", ""}
        df["service_count"] = (~normalized.isin(inactive)).sum(axis=1).astype(int)
    else:
        df["service_count"] = 0
    return df


def create_has_security_backup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag customers who have BOTH OnlineSecurity AND OnlineBackup.

    Rationale: Customers with both protection services are typically
    more engaged and security-conscious. They've invested in protecting
    their data, which correlates with lower churn — they value the service.
    """
    df = df.copy()

    def is_yes(col: str) -> pd.Series:
        if col in df.columns:
            return df[col].astype(str).str.strip().str.lower() == "yes"
        # Missing column → False for every row (index-aligned, unlike df.get
        # with an empty-Series default, which silently produces NaN)
        return pd.Series(False, index=df.index)

    df["has_security_backup"] = (is_yes("OnlineSecurity") & is_yes("OnlineBackup")).astype(int)
    return df


def create_high_value_short_tenure(
    df: pd.DataFrame,
    charge_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Identify high-revenue customers who are still in the early lifecycle.

    Rationale: These are the most DANGEROUS churners — they spend above
    the median but haven't built loyalty yet (tenure < 12 months).
    Losing them has the highest revenue impact. This feature helps
    the model prioritize retention efforts on the most valuable at-risk segment.

    Args:
        df: Input DataFrame.
        charge_threshold: MonthlyCharges cutoff. Must come from TRAINING
            data statistics in production (see FeatureEngineer) — computing
            it from the scored data itself would make the feature meaningless
            for single-row predictions. If None, the median of `df` is used
            (acceptable only for EDA on the full dataset).
    """
    df = df.copy()
    if charge_threshold is None:
        charge_threshold = df["MonthlyCharges"].median()
    df["high_value_short_tenure"] = (
        (df["MonthlyCharges"] > charge_threshold) & (df["tenure"] < 12)
    ).astype(int)
    return df


def create_charge_tenure_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the ratio of monthly charges to tenure.

    Rationale: High monthly charge relative to tenure suggests the
    customer is paying a premium without having built loyalty.
    This captures price sensitivity in the context of relationship duration.
    """
    df = df.copy()
    df["charge_tenure_ratio"] = df["MonthlyCharges"] / df["tenure"].clip(lower=1)
    return df


def engineer_features(
    df: pd.DataFrame,
    charge_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    For EDA and notebook use. Production code should use the
    FeatureEngineer transformer inside the model pipeline instead,
    so the MonthlyCharges threshold is learned from training data.

    Args:
        df: Cleaned DataFrame (after preprocessing.clean_data).
        charge_threshold: Optional fixed MonthlyCharges cutoff for
            high_value_short_tenure. Defaults to the median of `df`.

    Returns:
        DataFrame with all new features added.
    """
    df = create_tenure_group(df)
    df = create_avg_monthly_charge(df)
    df = create_service_count(df)
    df = create_has_security_backup(df)
    df = create_high_value_short_tenure(df, charge_threshold=charge_threshold)
    df = create_charge_tenure_ratio(df)

    new_features = ENGINEERED_CATEGORICAL + ENGINEERED_NUMERIC + ENGINEERED_FLAGS
    # DEBUG level: this runs on every pipeline transform (hundreds of times
    # during hyperparameter search), so INFO would flood the logs
    logger.debug(
        f"Feature engineering complete. Added {len(new_features)} features: "
        f"{sorted(new_features)}"
    )
    return df


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Sklearn transformer that adds all engineered features.

    Designed to be the FIRST step of the model pipeline:

        Pipeline([
            ("features", FeatureEngineer()),
            ("preprocessor", ColumnTransformer(...)),
            ("classifier", ...),
        ])

    `fit` learns the MonthlyCharges median from the training fold only,
    so cross-validation is leakage-free and single-row API requests get
    the exact same feature definitions as training data.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":
        """Learn training-data statistics needed by the features."""
        if "MonthlyCharges" not in X.columns:
            raise ValueError("FeatureEngineer requires a 'MonthlyCharges' column.")
        X = coerce_numeric_columns(X)
        self.monthly_charges_median_ = float(X["MonthlyCharges"].median())
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Add all engineered features using statistics learned in fit."""
        X = coerce_numeric_columns(X)
        return engineer_features(X, charge_threshold=self.monthly_charges_median_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return input feature names plus the engineered feature names."""
        if input_features is None:
            input_features = self.feature_names_in_
        # Same order in which transform() appends the new columns
        engineered = [
            "tenure_group", "avg_monthly_charge", "service_count",
            "has_security_backup", "high_value_short_tenure", "charge_tenure_ratio",
        ]
        new = [f for f in engineered if f not in list(input_features)]
        return np.concatenate([np.asarray(input_features, dtype=object), np.asarray(new, dtype=object)])
