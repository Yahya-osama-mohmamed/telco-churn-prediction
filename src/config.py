"""
Project Configuration — Central source of truth for all constants, paths, and settings.

This module defines:
- File paths (data, models, figures, reports)
- Random seed for reproducibility
- Feature column definitions
- Hyperparameter search spaces
- Model configuration
"""

from pathlib import Path
from typing import Dict, List, Any


# =============================================================================
# Project Root & Directory Paths
# =============================================================================
# All paths are relative to project root for portability
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for _dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR,
             REPORTS_DIR, LOGS_DIR, NOTEBOOKS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset Configuration
# =============================================================================
# Direct download URL — no Kaggle credentials required
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
PROCESSED_TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
PROCESSED_VAL_FILE = PROCESSED_DATA_DIR / "validation.csv"
PROCESSED_TEST_FILE = PROCESSED_DATA_DIR / "test.csv"


# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.15       # 15% for test
VAL_SIZE = 0.176       # 15% of total = 17.6% of remaining 85%


# =============================================================================
# Target Variable
# =============================================================================
TARGET = "Churn"
ID_COLUMN = "customerID"


# =============================================================================
# Feature Definitions
# =============================================================================
# Binary categorical features (Yes/No)
BINARY_FEATURES: List[str] = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling",
]

# Ordinal / special binary
SPECIAL_BINARY: List[str] = ["SeniorCitizen"]  # Already 0/1

# Multi-class categorical features
MULTICLASS_FEATURES: List[str] = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaymentMethod",
]

# Numeric features
NUMERIC_FEATURES: List[str] = [
    "tenure", "MonthlyCharges", "TotalCharges",
]

# Service columns — used in feature engineering (service_count)
SERVICE_COLUMNS: List[str] = [
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Engineered features (created by src.feature_engineering.FeatureEngineer)
ENGINEERED_NUMERIC: List[str] = [
    "avg_monthly_charge", "service_count", "charge_tenure_ratio",
]
ENGINEERED_FLAGS: List[str] = [
    "has_security_backup", "high_value_short_tenure",
]
ENGINEERED_CATEGORICAL: List[str] = ["tenure_group"]

# 0/1 flags that pass through the preprocessor unscaled
FLAG_FEATURES: List[str] = SPECIAL_BINARY + ENGINEERED_FLAGS


# =============================================================================
# Hyperparameter Search Spaces
# =============================================================================
LOGISTIC_REGRESSION_PARAMS: Dict[str, Any] = {
    "classifier__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    "classifier__penalty": ["l2"],
    "classifier__solver": ["lbfgs", "liblinear"],
    "classifier__max_iter": [1000],
}

XGBOOST_PARAMS: Dict[str, Any] = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [3, 4, 5, 6, 7],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_weight": [1, 3, 5, 7],
    "classifier__gamma": [0, 0.1, 0.2, 0.3],
    "classifier__reg_alpha": [0, 0.01, 0.1, 1.0],
    "classifier__reg_lambda": [0.5, 1.0, 1.5, 2.0],
}

LIGHTGBM_PARAMS: Dict[str, Any] = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [3, 5, 7, -1],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__num_leaves": [15, 31, 50, 70],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "classifier__min_child_samples": [5, 10, 20, 30],
    "classifier__reg_alpha": [0, 0.01, 0.1, 1.0],
    "classifier__reg_lambda": [0.5, 1.0, 1.5, 2.0],
}


# =============================================================================
# Model Training Configuration
# =============================================================================
CV_FOLDS = 5
N_ITER_SEARCH = 50          # RandomizedSearchCV iterations
SCORING_METRIC = "roc_auc"  # Primary metric for tuning


# =============================================================================
# MLflow Configuration
# =============================================================================
MLFLOW_EXPERIMENT_NAME = "telco-churn-prediction"
MLFLOW_TRACKING_URI = f"file:///{(PROJECT_ROOT / 'mlruns').as_posix()}"


# =============================================================================
# Model Persistence
# =============================================================================
FINAL_PIPELINE_PATH = MODELS_DIR / "final_pipeline.joblib"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.joblib"
PREPROCESSING_PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
