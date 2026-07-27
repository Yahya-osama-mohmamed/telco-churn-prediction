"""
Main Pipeline — CLI entry point for the complete ML workflow.

Orchestrates the entire pipeline from data loading to model saving:
1. Download and load data
2. Clean and preprocess
3. Engineer features
4. Split into train/val/test
5. Select features
6. Build preprocessing pipeline
7. Train and tune models
8. Evaluate and compare
9. Run SHAP explainability
10. Track experiments with MLflow
11. Save the best model

Usage:
    python main.py              # Run full pipeline
    python main.py --skip-mlflow  # Skip MLflow tracking
"""

import json
import time
import argparse
import warnings
import datetime
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress convergence and future warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import (
    RANDOM_STATE, TARGET, FIGURES_DIR, MODELS_DIR,
    NUMERIC_FEATURES, BINARY_FEATURES, MULTICLASS_FEATURES,
    ENGINEERED_NUMERIC, ENGINEERED_CATEGORICAL, FLAG_FEATURES,
    FINAL_PIPELINE_PATH, BEST_MODEL_PATH, FEATURE_NAMES_PATH,
    PREPROCESSING_PIPELINE_PATH, MODEL_METADATA_PATH,
    MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
)
from src.logger import get_logger
from src.data_loader import load_and_validate
from src.preprocessing import clean_data, split_data, build_preprocessing_pipeline
from src.feature_engineering import engineer_features, FeatureEngineer
from src.feature_selection import select_features
from src.model_training import train_all_models
from src.model_evaluation import evaluate_all_models, tune_decision_threshold
from src.explainability import run_explainability
from sklearn.pipeline import Pipeline

logger = get_logger(__name__)


# =============================================================================
# EDA Visualizations
# =============================================================================

def run_eda(df: pd.DataFrame) -> None:
    """
    Generate all EDA visualizations and save to figures/.

    Args:
        df: Cleaned DataFrame with target column.
    """
    logger.info("=" * 60)
    logger.info("EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 60)

    plt.style.use("seaborn-v0_8-whitegrid")

    # 1. Churn Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = df[TARGET].value_counts()
    colors = ["#2196F3", "#FF5722"]

    axes[0].bar(["No Churn", "Churn"], churn_counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Churn Distribution", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("Count")
    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 50, str(v), ha="center", fontweight="bold")

    axes[1].pie(
        churn_counts.values, labels=["No Churn", "Churn"],
        autopct="%1.1f%%", colors=colors, startangle=90,
        textprops={"fontsize": 12},
    )
    axes[1].set_title("Churn Percentage", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_churn_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Churn by Contract Type
    if "Contract" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ct = pd.crosstab(df["Contract"], df[TARGET], normalize="index") * 100
        ct.columns = ["No Churn %", "Churn %"]
        ct.plot(kind="bar", stacked=True, color=colors, ax=ax, edgecolor="white")
        ax.set_title("Churn Rate by Contract Type", fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        ax.legend(loc="upper right")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_churn_by_contract.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Churn by Internet Service
    if "InternetService" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ct = pd.crosstab(df["InternetService"], df[TARGET], normalize="index") * 100
        ct.columns = ["No Churn %", "Churn %"]
        ct.plot(kind="bar", stacked=True, color=colors, ax=ax, edgecolor="white")
        ax.set_title("Churn Rate by Internet Service", fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_churn_by_internet.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Tenure Distribution by Churn
    fig, ax = plt.subplots(figsize=(10, 5))
    for churn_val, label, color in [(0, "No Churn", "#2196F3"), (1, "Churn", "#FF5722")]:
        subset = df[df[TARGET] == churn_val]["tenure"]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_title("Tenure Distribution by Churn Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_tenure_by_churn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Monthly Charges Distribution by Churn
    fig, ax = plt.subplots(figsize=(10, 5))
    for churn_val, label, color in [(0, "No Churn", "#2196F3"), (1, "Churn", "#FF5722")]:
        subset = df[df[TARGET] == churn_val]["MonthlyCharges"]
        ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="white")
    ax.set_title("Monthly Charges Distribution by Churn", fontsize=13, fontweight="bold")
    ax.set_xlabel("Monthly Charges ($)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_monthly_charges_by_churn.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 6. Correlation Heatmap
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=ax,
            square=True, linewidths=0.5,
            vmin=-1, vmax=1,
        )
        ax.set_title("Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 7. Churn by Payment Method
    if "PaymentMethod" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        ct = pd.crosstab(df["PaymentMethod"], df[TARGET], normalize="index") * 100
        ct.columns = ["No Churn %", "Churn %"]
        ct.plot(kind="bar", stacked=True, color=colors, ax=ax, edgecolor="white")
        ax.set_title("Churn Rate by Payment Method", fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        plt.xticks(rotation=15)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_churn_by_payment.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 8. Box Plots for Outlier Detection
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(["tenure", "MonthlyCharges", "TotalCharges"]):
        if col in df.columns:
            box_data = [df[df[TARGET] == 0][col], df[df[TARGET] == 1][col]]
            bp = axes[i].boxplot(box_data, tick_labels=["No Churn", "Churn"], patch_artist=True)
            bp["boxes"][0].set_facecolor("#2196F3")
            bp["boxes"][1].set_facecolor("#FF5722")
            axes[i].set_title(col, fontsize=12, fontweight="bold")
            axes[i].set_ylabel(col)
    plt.suptitle("Box Plots — Outlier Detection", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 9. Churn by Senior Citizen
    if "SeniorCitizen" in df.columns:
        fig, ax = plt.subplots(figsize=(6, 5))
        ct = pd.crosstab(df["SeniorCitizen"], df[TARGET], normalize="index") * 100
        ct.columns = ["No Churn %", "Churn %"]
        ct.index = ["Non-Senior", "Senior"]
        ct.plot(kind="bar", stacked=True, color=colors, ax=ax, edgecolor="white")
        ax.set_title("Churn Rate by Senior Citizen Status", fontsize=13, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_xlabel("")
        plt.xticks(rotation=0)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_churn_by_senior.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 10. Stacked Bar: Services vs Churn
    service_cols = ["OnlineSecurity", "TechSupport", "OnlineBackup",
                    "DeviceProtection", "StreamingTV", "StreamingMovies"]
    available_services = [c for c in service_cols if c in df.columns]
    if available_services:
        fig, ax = plt.subplots(figsize=(12, 6))
        churn_rates = {}
        for svc in available_services:
            rate = df.groupby(svc)[TARGET].mean() * 100
            for val, pct in rate.items():
                churn_rates[f"{svc}\n({val})"] = pct

        sorted_items = sorted(churn_rates.items(), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_items)
        bar_colors = ["#FF5722" if v > 30 else "#2196F3" for v in values]
        ax.barh(labels, values, color=bar_colors, edgecolor="white")
        ax.set_xlabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Service Subscription", fontsize=13, fontweight="bold")
        ax.axvline(x=df[TARGET].mean() * 100, color="black", linestyle="--", alpha=0.5, label="Average")
        ax.legend()
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "eda_churn_by_services.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    logger.info(f"EDA complete. Saved {len(list(FIGURES_DIR.glob('eda_*.png')))} visualizations to {FIGURES_DIR}")


# =============================================================================
# MLflow Tracking
# =============================================================================

def log_to_mlflow(
    model_results: dict,
    comparison_df: pd.DataFrame,
    best_model_name: str,
    skip_mlflow: bool = False,
) -> None:
    """
    Log all experiment results to MLflow.

    Args:
        model_results: Training results for all models.
        comparison_df: Model comparison DataFrame.
        best_model_name: Name of the best model.
        skip_mlflow: If True, skip MLflow logging entirely.
    """
    if skip_mlflow:
        logger.info("MLflow tracking skipped (--skip-mlflow flag).")
        return

    try:
        import mlflow
        import mlflow.sklearn

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        for name, result in model_results.items():
            with mlflow.start_run(run_name=name):
                # Log parameters
                for param_key, param_val in result["best_params"].items():
                    mlflow.log_param(param_key, param_val)

                # Log metrics from comparison table
                model_row = comparison_df[comparison_df["Model"] == name].iloc[0]
                mlflow.log_metric("test_accuracy", model_row["Test Accuracy"])
                mlflow.log_metric("test_precision", model_row["Test Precision"])
                mlflow.log_metric("test_recall", model_row["Test Recall"])
                mlflow.log_metric("test_f1", model_row["Test F1"])
                mlflow.log_metric("test_roc_auc", model_row["Test ROC-AUC"])
                mlflow.log_metric("val_roc_auc", model_row["Val ROC-AUC"])
                mlflow.log_metric("train_roc_auc", model_row["Train ROC-AUC"])
                mlflow.log_metric("overfit_gap", model_row["Overfit Gap"])
                mlflow.log_metric("training_time", result["train_time"])

                # Tag as best model if applicable
                if name == best_model_name:
                    mlflow.set_tag("best_model", "true")

                # Log the model
                mlflow.sklearn.log_model(result["pipeline"], "model")

                # Log figures only on the best model's run — shared EDA and
                # comparison plots attached to every run would misattribute
                # model-specific artifacts (SHAP, confusion matrices)
                if name == best_model_name:
                    for fig_path in FIGURES_DIR.glob("*.png"):
                        mlflow.log_artifact(str(fig_path), "figures")

                logger.info(f"MLflow: Logged run for {name}")

        logger.info("MLflow tracking complete. Run 'mlflow ui' to view experiments.")

    except Exception as e:
        logger.warning(f"MLflow tracking failed: {e}. Continuing without MLflow.")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_pipeline(skip_mlflow: bool = False) -> None:
    """
    Execute the complete ML pipeline end-to-end.

    This is the main orchestration function that calls all modules
    in the correct order and handles the flow of data between them.
    """
    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("TELCO CUSTOMER CHURN PREDICTION PIPELINE")
    logger.info("=" * 60)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logger.info("\n📥 Step 1: Loading data...")
    df_raw, profile = load_and_validate()
    logger.info(f"Dataset: {profile['n_rows']} rows × {profile['n_cols']} columns")
    logger.info(f"Class distribution: {profile['class_distribution']}")
    logger.info(f"Missing values: {profile['n_missing_total']} total")

    # =========================================================================
    # Step 2: Clean Data
    # =========================================================================
    logger.info("\n🧹 Step 2: Cleaning data...")
    df_clean = clean_data(df_raw)

    # =========================================================================
    # Step 3: EDA Visualizations
    # =========================================================================
    logger.info("\n📊 Step 3: Running EDA...")
    run_eda(df_clean)

    # =========================================================================
    # Step 4: Feature Engineering (reference artifact for EDA/notebooks)
    # =========================================================================
    logger.info("\n🔧 Step 4: Engineering features (reference artifact)...")
    # In the model pipelines, feature engineering happens INSIDE the sklearn
    # Pipeline (FeatureEngineer step) so its statistics are learned from
    # training folds only. This full-data version is saved purely for
    # EDA/notebook reference.
    df_featured = engineer_features(df_clean)
    df_featured.to_csv(PROCESSED_DATA_DIR / "featured_data.csv", index=False)
    logger.info(f"Saved featured data: {df_featured.shape}")

    # =========================================================================
    # Step 5: Split Data (Train / Validation / Test) — raw features
    # =========================================================================
    logger.info("\n✂️ Step 5: Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    # =========================================================================
    # Step 6: Feature Selection (analysis)
    # =========================================================================
    logger.info("\n🎯 Step 6: Analyzing feature importance...")

    # Transform training data exactly the way the models will see it
    analysis_pipeline = Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", build_preprocessing_pipeline(
            numeric_features=NUMERIC_FEATURES + ENGINEERED_NUMERIC,
            binary_features=BINARY_FEATURES,
            multiclass_features=MULTICLASS_FEATURES + ENGINEERED_CATEGORICAL,
            flag_features=FLAG_FEATURES,
        )),
    ])
    X_train_transformed = analysis_pipeline.fit_transform(X_train, y_train)
    feature_names_out = list(
        analysis_pipeline.named_steps["preprocessor"].get_feature_names_out()
    )
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names_out)

    selected_features, rankings = select_features(
        X_train_df, y_train.reset_index(drop=True), top_n=15,
    )
    # Analysis only: tree models handle weak features well and n >> p here,
    # so all features go to the models. The rankings and figures document
    # which features carry the signal.
    logger.info(
        f"Feature-importance analysis found {len(selected_features)} strong "
        f"features (union of MI + RF top-15). Models are trained on the full "
        f"feature set; see figures/feature_selection_*.png for the rankings."
    )

    # =========================================================================
    # Step 7: Build Final Preprocessing Pipeline
    # =========================================================================
    logger.info("\n🏗️ Step 7: Building final preprocessing pipeline...")

    final_preprocessor = build_preprocessing_pipeline(
        numeric_features=NUMERIC_FEATURES + ENGINEERED_NUMERIC,
        binary_features=BINARY_FEATURES,
        multiclass_features=MULTICLASS_FEATURES + ENGINEERED_CATEGORICAL,
        flag_features=FLAG_FEATURES,
    )

    # =========================================================================
    # Step 8: Train Models
    # =========================================================================
    logger.info("\n🚀 Step 8: Training models...")
    model_results = train_all_models(final_preprocessor, X_train, y_train)

    # =========================================================================
    # Step 9: Evaluate & Compare Models
    # =========================================================================
    logger.info("\n📈 Step 9: Evaluating models...")
    comparison_df = evaluate_all_models(
        model_results, X_train, y_train, X_val, y_val, X_test, y_test,
    )

    # Determine best model (comparison_df is ranked by Validation ROC-AUC)
    best_model_name = comparison_df.iloc[0]["Model"]
    best_pipeline = model_results[best_model_name]["pipeline"]
    logger.info(f"\n🏆 Best model: {best_model_name}")

    # Tune the decision threshold on the validation set (default 0.5 is
    # rarely optimal with ~27% churn prevalence)
    decision_threshold = tune_decision_threshold(best_pipeline, X_val, y_val)

    # Evaluate the DEPLOYED operating point: test metrics at the tuned
    # threshold (the comparison table uses the default 0.5 cutoff)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= decision_threshold).astype(int)
    test_at_threshold = {
        "accuracy": round(float(accuracy_score(y_test, test_preds)), 4),
        "precision": round(float(precision_score(y_test, test_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, test_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, test_preds, zero_division=0)), 4),
    }
    logger.info(
        f"Test metrics at deployed threshold {decision_threshold:.3f}: "
        f"{test_at_threshold}"
    )

    # =========================================================================
    # Step 10: SHAP Explainability
    # =========================================================================
    logger.info("\n🔍 Step 10: Running SHAP explainability...")
    shap_results = run_explainability(
        best_pipeline, X_train, X_test, model_name=best_model_name,
    )

    # =========================================================================
    # Step 11: MLflow Experiment Tracking
    # =========================================================================
    logger.info("\n📋 Step 11: Logging to MLflow...")
    log_to_mlflow(model_results, comparison_df, best_model_name, skip_mlflow)

    # =========================================================================
    # Step 12: Save Models
    # =========================================================================
    logger.info("\n💾 Step 12: Saving models...")

    # Save complete pipeline (preprocessor + best model)
    joblib.dump(best_pipeline, FINAL_PIPELINE_PATH)
    logger.info(f"Saved final pipeline: {FINAL_PIPELINE_PATH}")

    # Save just the model (for advanced use)
    best_classifier = best_pipeline.named_steps["classifier"]
    joblib.dump(best_classifier, BEST_MODEL_PATH)
    logger.info(f"Saved best model: {BEST_MODEL_PATH}")

    # Save feature names (raw input columns the pipeline expects)
    feature_names = list(X_train.columns)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info(f"Saved feature names: {FEATURE_NAMES_PATH}")

    # Save standalone preprocessor
    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    joblib.dump(preprocessor_fitted, PREPROCESSING_PIPELINE_PATH)
    logger.info(f"Saved preprocessor: {PREPROCESSING_PIPELINE_PATH}")

    # Save model metadata (consumed by the API for threshold + versioning)
    metadata = {
        "best_model": best_model_name,
        "decision_threshold": round(decision_threshold, 4),
        "val_roc_auc": round(float(comparison_df.iloc[0]["Val ROC-AUC"]), 4),
        "test_roc_auc": round(float(comparison_df.iloc[0]["Test ROC-AUC"]), 4),
        "test_metrics_at_threshold": test_at_threshold,
        "trained_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "n_training_rows": int(len(X_train)),
    }
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved model metadata: {MODEL_METADATA_PATH}")

    # Save processed splits
    X_train.assign(**{TARGET: y_train}).to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    X_val.assign(**{TARGET: y_val}).to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    X_test.assign(**{TARGET: y_test}).to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    logger.info("Saved processed train/val/test splits.")

    # =========================================================================
    # Summary
    # =========================================================================
    total_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {total_time:.1f}s")
    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Test ROC-AUC: {comparison_df.iloc[0]['Test ROC-AUC']:.4f}")
    logger.info(f"Figures saved: {len(list(FIGURES_DIR.glob('*.png')))}")
    logger.info(f"Models saved: {len(list(MODELS_DIR.glob('*.joblib')))}")
    logger.info(f"\nProject artifacts in: {MODELS_DIR.parent}")


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Telco Customer Churn Prediction Pipeline",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Skip MLflow experiment tracking.",
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)
    run_pipeline(skip_mlflow=args.skip_mlflow)
