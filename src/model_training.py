"""
Model Training — Train, tune, and cross-validate ML models.

Models:
1. Logistic Regression — interpretable baseline
2. XGBoost — primary gradient boosting model
3. LightGBM — fast gradient boosting alternative

All models are wrapped in sklearn Pipelines with the preprocessing
ColumnTransformer to ensure consistent transformations and prevent
data leakage during cross-validation.
"""

import time
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import lightgbm as lgb

from src.config import (
    RANDOM_STATE, CV_FOLDS, N_ITER_SEARCH, SCORING_METRIC,
    LOGISTIC_REGRESSION_PARAMS, XGBOOST_PARAMS, LIGHTGBM_PARAMS,
)
from src.feature_engineering import FeatureEngineer
from src.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Model Definitions
# =============================================================================

def get_model_configs(
    scale_pos_weight: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Define all model configurations with their classifiers and param grids.

    Args:
        scale_pos_weight: Ratio of negative to positive class for
                          imbalanced dataset handling. Calculated as
                          n_negative / n_positive.

    Returns:
        Dictionary mapping model names to their config dicts containing:
        - 'classifier': sklearn-compatible classifier instance
        - 'params': hyperparameter search space
        - 'description': human-readable model description
    """
    configs = {
        "Logistic Regression": {
            "classifier": LogisticRegression(
                random_state=RANDOM_STATE,
                class_weight="balanced",  # Handles class imbalance
                max_iter=1000,
            ),
            "params": LOGISTIC_REGRESSION_PARAMS,
            "description": (
                "Interpretable linear baseline. Uses L2 regularization "
                "and balanced class weights to handle imbalanced data."
            ),
        },
        "XGBoost": {
            "classifier": xgb.XGBClassifier(
                random_state=RANDOM_STATE,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                verbosity=0,
                # Single-threaded: RandomizedSearchCV already parallelizes across
                # processes; nested thread pools oversubscribe cores and can get
                # loky workers OOM-killed on Windows.
                n_jobs=1,
            ),
            "params": XGBOOST_PARAMS,
            "description": (
                "Gradient boosting with regularization. Uses scale_pos_weight "
                "for class imbalance and L1/L2 regularization to prevent overfitting."
            ),
        },
        "LightGBM": {
            "classifier": lgb.LGBMClassifier(
                random_state=RANDOM_STATE,
                is_unbalance=True,  # Handles class imbalance
                verbosity=-1,
                n_jobs=1,  # See XGBoost note: search-level parallelism only
            ),
            "params": LIGHTGBM_PARAMS,
            "description": (
                "Fast gradient boosting with leaf-wise growth. Uses is_unbalance "
                "for class imbalance and histogram-based splitting for speed."
            ),
        },
    }

    return configs


# =============================================================================
# Training Pipeline
# =============================================================================

def build_model_pipeline(
    preprocessor: ColumnTransformer,
    classifier: Any,
) -> Pipeline:
    """
    Create an sklearn Pipeline: feature engineering → preprocessing → classifier.

    The pipeline ensures that:
    - Engineered-feature statistics (MonthlyCharges median) and all
      preprocessing are fit ONLY on training data
    - Cross-validation properly re-fits every step per fold
    - The serialized pipeline is fully self-contained for deployment:
      it accepts raw customer records, no manual feature engineering needed

    Args:
        preprocessor: Unfitted ColumnTransformer (cloned per model).
        classifier: sklearn-compatible classifier instance.

    Returns:
        sklearn Pipeline.
    """
    return Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", clone(preprocessor)),
        ("classifier", classifier),
    ])


def train_single_model(
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = CV_FOLDS,
    n_iter: int = N_ITER_SEARCH,
    scoring: str = SCORING_METRIC,
) -> Tuple[Pipeline, Dict[str, Any], float]:
    """
    Train a single model with hyperparameter tuning via RandomizedSearchCV.

    Uses stratified K-fold cross-validation to ensure each fold has
    the same class distribution as the full training set.

    Args:
        pipeline: sklearn Pipeline with preprocessor + classifier.
        param_grid: Hyperparameter search space.
        X_train: Training features.
        y_train: Training labels.
        cv_folds: Number of CV folds.
        n_iter: Number of random parameter combinations to try.
        scoring: Metric to optimize.

    Returns:
        Tuple of (best pipeline, best params, training time in seconds).
    """
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=min(n_iter, _count_param_combinations(param_grid)),
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )

    start_time = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - start_time

    best_params = search.best_params_
    best_score = search.best_score_

    logger.info(
        f"Best CV {scoring}: {best_score:.4f} "
        f"(training time: {train_time:.1f}s)"
    )
    logger.info(f"Best params: {best_params}")

    return search.best_estimator_, best_params, train_time


def _count_param_combinations(param_grid: Dict[str, Any]) -> int:
    """Count the total number of parameter combinations in a grid."""
    total = 1
    for values in param_grid.values():
        if isinstance(values, list):
            total *= len(values)
    return total


def train_all_models(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, Dict[str, Any]]:
    """
    Train all models with hyperparameter tuning.

    This is the main entry point for model training. It:
    1. Calculates class imbalance ratio
    2. Builds pipelines for each model
    3. Runs RandomizedSearchCV for each
    4. Returns all results in a structured dictionary

    Args:
        preprocessor: ColumnTransformer for feature transformation.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        Dictionary mapping model names to result dicts containing:
        - 'pipeline': Best fitted pipeline
        - 'best_params': Best hyperparameters
        - 'train_time': Training time in seconds
        - 'description': Model description
    """
    # Calculate class imbalance ratio for XGBoost
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive
    logger.info(
        f"Class imbalance: {n_negative} negative / {n_positive} positive "
        f"(ratio: {scale_pos_weight:.2f})"
    )

    model_configs = get_model_configs(scale_pos_weight=scale_pos_weight)
    results = {}

    for name, config in model_configs.items():
        logger.info("=" * 60)
        logger.info(f"Training: {name}")
        logger.info("=" * 60)

        pipeline = build_model_pipeline(
            preprocessor=preprocessor,
            classifier=config["classifier"],
        )

        best_pipeline, best_params, train_time = train_single_model(
            pipeline=pipeline,
            param_grid=config["params"],
            X_train=X_train,
            y_train=y_train,
        )

        results[name] = {
            "pipeline": best_pipeline,
            "best_params": best_params,
            "train_time": train_time,
            "description": config["description"],
        }

    logger.info("=" * 60)
    logger.info("All models trained successfully.")
    logger.info("=" * 60)

    return results
