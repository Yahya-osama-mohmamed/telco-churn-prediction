"""
Data Preprocessing Pipeline — Cleaning, Encoding, and Scaling.

Implements a scikit-learn Pipeline + ColumnTransformer approach to ensure:
- No data leakage (transformers fit on training data only)
- Reproducible, consistent transformations
- Easy serialization for deployment

Steps:
1. Handle missing values (TotalCharges)
2. Drop customerID
3. Encode binary categoricals (Yes/No → 1/0)
4. One-hot encode multi-class categoricals
5. Standard-scale numeric features
6. Pass 0/1 flag columns through unchanged; drop everything else
   (so unexpected columns like customerID in a batch CSV can't
   reach the model)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.config import (
    RANDOM_STATE, TEST_SIZE, VAL_SIZE, TARGET, ID_COLUMN,
    BINARY_FEATURES, MULTICLASS_FEATURES, NUMERIC_FEATURES, FLAG_FEATURES,
    ENGINEERED_CATEGORICAL, ENGINEERED_NUMERIC,
)
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Custom Transformers
# =============================================================================

class BinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Encode binary Yes/No and Male/Female columns to 1/0.

    This is preferred over OneHotEncoder for binary features because:
    - It produces a single column (not two dummy columns)
    - It's more memory-efficient
    - It preserves interpretability

    Values are stripped of surrounding whitespace before mapping.
    Unmapped values raise a ValueError — failing fast beats silently
    feeding garbage into the model in production.
    """

    _MAPPING = {
        "Yes": 1, "No": 0,
        "Male": 1, "Female": 0,
    }

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "BinaryEncoder":
        """Record input feature names; the mapping itself is deterministic."""
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply binary encoding to all object-dtype columns."""
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == object:
                stripped = X[col].astype(str).str.strip()
                mapped = stripped.map(self._MAPPING)
                unknown = mapped.isna() & X[col].notna()
                if unknown.any():
                    bad_values = sorted(stripped[unknown].unique().tolist())
                    raise ValueError(
                        f"BinaryEncoder: column '{col}' contains values that are "
                        f"not binary Yes/No or Male/Female: {bad_values}"
                    )
                n_missing = int(X[col].isna().sum())
                if n_missing:
                    logger.warning(
                        f"BinaryEncoder: column '{col}' has {n_missing} missing "
                        f"values; encoding them as 0 ('No')."
                    )
                X[col] = mapped.fillna(0).astype(int)
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Binary encoding is one-to-one: output names equal input names."""
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return self.feature_names_in_


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Initial data cleaning steps applied before the main pipeline.

    Handles:
    - Removing duplicate customers (by customerID, while it still exists)
    - Dropping customerID (not a predictive feature)
    - Converting TotalCharges blanks to NaN → imputing with 0
      (these are new customers with tenure=0)
    - Encoding the target variable
    """

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DataCleaner":
        """No fitting required."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clean the DataFrame."""
        df = X.copy()

        # Remove duplicate customers by ID while the ID is still present.
        # (Deduplicating on all columns AFTER dropping the ID would delete
        # distinct customers who happen to share an identical profile.)
        if ID_COLUMN in df.columns:
            n_before = len(df)
            df = df.drop_duplicates(subset=[ID_COLUMN])
            n_dropped = n_before - len(df)
            if n_dropped > 0:
                logger.info(f"Dropped {n_dropped} duplicate customer IDs.")
            df = df.drop(columns=[ID_COLUMN])

        # Convert TotalCharges to numeric (blanks → NaN)
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"], errors="coerce"
            )
            # Impute NaN with 0 — these are new customers (tenure=0)
            df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

        # Encode target variable
        if TARGET in df.columns and df[TARGET].dtype == object:
            df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

        return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply initial cleaning to the raw DataFrame.

    This is a convenience function that wraps the DataCleaner transformer
    for use outside of the sklearn Pipeline (e.g., in EDA notebooks).

    Args:
        df: Raw DataFrame from data_loader.

    Returns:
        Cleaned DataFrame ready for feature engineering.
    """
    cleaner = DataCleaner()
    df_clean = cleaner.transform(df)
    logger.info(
        f"Cleaning complete: {df_clean.shape[0]} rows × {df_clean.shape[1]} cols"
    )
    return df_clean


# =============================================================================
# Train / Validation / Test Split
# =============================================================================

def split_data(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """
    Split data into Train (70%) / Validation (15%) / Test (15%).

    Uses stratified splitting to preserve class distribution across all sets.
    The split is done in two steps:
    1. Split into train+val (85%) and test (15%)
    2. Split train+val into train (70%) and val (15%)

    Args:
        df: Cleaned DataFrame with target column.
        target: Target column name.
        test_size: Fraction reserved for test set.
        val_size: Fraction of train+val reserved for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]

    # Step 1: Split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Step 2: Split remaining into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        stratify=y_temp,
        random_state=random_state,
    )

    logger.info(
        f"Data split: Train={len(X_train)} ({len(X_train)/len(df)*100:.1f}%), "
        f"Val={len(X_val)} ({len(X_val)/len(df)*100:.1f}%), "
        f"Test={len(X_test)} ({len(X_test)/len(df)*100:.1f}%)"
    )
    logger.info(
        f"Target distribution — "
        f"Train: {y_train.mean():.3f}, "
        f"Val: {y_val.mean():.3f}, "
        f"Test: {y_test.mean():.3f}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# Sklearn Preprocessing Pipeline
# =============================================================================

def build_preprocessing_pipeline(
    numeric_features: List[str],
    binary_features: List[str],
    multiclass_features: List[str],
    flag_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles all feature transformations.

    Architecture:
    - Numeric features → Impute missing → StandardScaler
    - Binary features → BinaryEncoder (Yes/No → 1/0)
    - Multi-class features → OneHotEncoder (drop first to avoid multicollinearity)
    - Flag features (already 0/1) → passthrough
    - Everything else → DROPPED. This is a deliberate safety net: extra
      columns in inference data (customerID, Churn, arbitrary CSV columns)
      are ignored instead of crashing or leaking into the model.

    This transformer is fit ONLY on training data to prevent data leakage.

    Args:
        numeric_features: List of numeric column names.
        binary_features: List of binary (Yes/No) column names.
        multiclass_features: List of multi-class categorical column names.
        flag_features: List of already-0/1 columns to pass through unchanged.

    Returns:
        Configured ColumnTransformer.
    """
    # Numeric pipeline: impute missing → scale
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Binary pipeline: encode Yes/No to 1/0
    binary_pipeline = Pipeline([
        ("encoder", BinaryEncoder()),
    ])

    # Multi-class pipeline: one-hot encode.
    # drop=None (not "first"): with drop="first" + handle_unknown="ignore",
    # an UNSEEN category at inference encodes as all-zeros — identical to the
    # dropped baseline category — so unknown values would silently score as
    # the baseline class. With drop=None, unknowns get a distinct all-zeros
    # row no real category shares. The regularized/tree models used here
    # don't need the collinearity drop.
    multiclass_pipeline = Pipeline([
        ("encoder", OneHotEncoder(
            drop=None,
            sparse_output=False,    # Return dense array for compatibility
            handle_unknown="ignore",  # Handle unseen categories gracefully in production
        )),
    ])

    transformers = [
        ("num", numeric_pipeline, numeric_features),
        ("bin", binary_pipeline, binary_features),
        ("cat", multiclass_pipeline, multiclass_features),
    ]
    if flag_features:
        transformers.append(("flag", "passthrough", flag_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_default_preprocessor() -> ColumnTransformer:
    """
    Build the default preprocessor for the full engineered feature set.

    Returns:
        ColumnTransformer covering original + engineered features.
    """
    return build_preprocessing_pipeline(
        numeric_features=NUMERIC_FEATURES + ENGINEERED_NUMERIC,
        binary_features=BINARY_FEATURES,
        multiclass_features=MULTICLASS_FEATURES + ENGINEERED_CATEGORICAL,
        flag_features=FLAG_FEATURES,
    )
