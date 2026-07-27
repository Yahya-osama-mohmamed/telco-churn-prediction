"""
Model Evaluation — Comprehensive metrics, visualizations, and comparison.

Evaluates each model on Train, Validation, and Test sets using:
- Accuracy, Precision, Recall, F1 Score, ROC-AUC
- Confusion matrices
- ROC curves and Precision-Recall curves
- Overfitting analysis (train-val gap)
- Side-by-side comparison table
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
    precision_recall_curve, average_precision_score,
)
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, REPORTS_DIR
from src.logger import get_logger

logger = get_logger(__name__)

# Use a clean style for all plots
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    dataset_name: str = "Test",
) -> Dict[str, float]:
    """
    Compute all classification metrics for a single dataset split.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class.
        dataset_name: Name of the dataset split (for logging).

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    logger.info(
        f"{dataset_name} metrics — "
        f"Acc: {metrics['accuracy']:.4f}, "
        f"Prec: {metrics['precision']:.4f}, "
        f"Rec: {metrics['recall']:.4f}, "
        f"F1: {metrics['f1']:.4f}, "
        f"AUC: {metrics['roc_auc']:.4f}"
    )

    return metrics


def evaluate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
    model_name: str,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a single model on all three dataset splits.

    Args:
        pipeline: Fitted sklearn Pipeline.
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        model_name: Name for logging and labeling.

    Returns:
        Nested dict: {split_name: {metric_name: value}}.
    """
    logger.info(f"\n--- Evaluating: {model_name} ---")
    results = {}

    for split_name, X, y in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = pipeline.predict(X)
        y_prob = pipeline.predict_proba(X)[:, 1]
        results[split_name] = compute_metrics(y, y_pred, y_prob, split_name.title())

    # Overfitting check: train vs validation gap
    train_auc = results["train"]["roc_auc"]
    val_auc = results["validation"]["roc_auc"]
    gap = train_auc - val_auc
    if gap > 0.05:
        logger.warning(
            f"⚠ Possible overfitting: Train AUC ({train_auc:.4f}) - "
            f"Val AUC ({val_auc:.4f}) = {gap:.4f}"
        )
    else:
        logger.info(
            f"✓ No significant overfitting: "
            f"Train-Val AUC gap = {gap:.4f}"
        )

    return results


# =============================================================================
# Visualizations
# =============================================================================

def plot_confusion_matrix(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    dataset_name: str = "Test",
) -> None:
    """
    Plot a confusion matrix heatmap.

    Args:
        pipeline: Fitted Pipeline.
        X: Features.
        y: True labels.
        model_name: Model name for the title.
        dataset_name: Dataset split name.
    """
    y_pred = pipeline.predict(X)
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax, cbar=False,
        annot_kws={"size": 14},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {model_name} ({dataset_name})",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{dataset_name.lower()}.png"
    fig.savefig(FIGURES_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    model_results: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Plot ROC curves for all models on the test set (overlay).

    This allows direct visual comparison of discriminative ability.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for idx, (name, result) in enumerate(model_results.items()):
        pipeline = result["pipeline"]
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        ax.plot(
            fpr, tpr,
            label=f"{name} (AUC = {auc:.4f})",
            color=colors[idx % len(colors)],
            linewidth=2,
        )

    # Diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "roc_curves_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curves comparison plot.")


def plot_precision_recall_curves(
    model_results: Dict[str, Dict[str, Any]],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """
    Plot Precision-Recall curves for all models on the test set.

    PR curves are more informative than ROC when classes are imbalanced
    because they focus on the performance for the minority class.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]

    for idx, (name, result) in enumerate(model_results.items()):
        pipeline = result["pipeline"]
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        ax.plot(
            recall, precision,
            label=f"{name} (AP = {ap:.4f})",
            color=colors[idx % len(colors)],
            linewidth=2,
        )

    # Baseline: prevalence of positive class
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "pr_curves_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Precision-Recall curves comparison plot.")


# =============================================================================
# Model Comparison Table
# =============================================================================

def create_comparison_table(
    all_metrics: Dict[str, Dict[str, Dict[str, float]]],
    training_times: Dict[str, float],
    descriptions: Dict[str, str],
) -> pd.DataFrame:
    """
    Create a comprehensive model comparison table.

    Columns include performance metrics on all splits, training time,
    and overfitting gap (train AUC - validation AUC).

    Args:
        all_metrics: {model_name: {split: {metric: value}}}.
        training_times: {model_name: seconds}.
        descriptions: {model_name: description string}.

    Returns:
        DataFrame with one row per model, sorted by validation ROC-AUC
        (the selection metric; test columns are reported but never used
        to pick the winner).
    """
    rows = []
    for name in all_metrics:
        test = all_metrics[name].get("test", {})
        val = all_metrics[name].get("validation", {})
        train = all_metrics[name].get("train", {})

        row = {
            "Model": name,
            "Test Accuracy": test.get("accuracy", 0),
            "Test Precision": test.get("precision", 0),
            "Test Recall": test.get("recall", 0),
            "Test F1": test.get("f1", 0),
            "Test ROC-AUC": test.get("roc_auc", 0),
            "Val ROC-AUC": val.get("roc_auc", 0),
            "Train ROC-AUC": train.get("roc_auc", 0),
            "Overfit Gap": train.get("roc_auc", 0) - val.get("roc_auc", 0),
            "Training Time (s)": training_times.get(name, 0),
            "Description": descriptions.get(name, ""),
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows)
    # Rank by VALIDATION ROC-AUC: choosing the winner on the test set would
    # leak test information into model selection and bias the reported score.
    comparison_df = comparison_df.sort_values("Val ROC-AUC", ascending=False)
    comparison_df = comparison_df.reset_index(drop=True)

    # Save to CSV
    comparison_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    logger.info(f"Model comparison table saved to {REPORTS_DIR / 'model_comparison.csv'}")

    # Pretty print
    display_cols = [
        "Model", "Test Accuracy", "Test Precision", "Test Recall",
        "Test F1", "Test ROC-AUC", "Val ROC-AUC", "Overfit Gap", "Training Time (s)",
    ]
    logger.info("\n" + comparison_df[display_cols].to_string(index=False))

    return comparison_df


def plot_comparison_bar_chart(comparison_df: pd.DataFrame) -> None:
    """
    Create a grouped bar chart comparing key metrics across models.
    """
    metrics = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC"]
    models = comparison_df["Model"].tolist()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.25
    offsets = np.linspace(-width, width, len(models))

    colors = ["#2196F3", "#FF5722", "#4CAF50"]
    for idx, model in enumerate(models):
        values = comparison_df[comparison_df["Model"] == model][metrics].values[0]
        ax.bar(
            x + offsets[idx], values,
            width=width, label=model,
            color=colors[idx % len(colors)],
            alpha=0.85, edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("Test ", "") for m in metrics], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.05])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fig.savefig(FIGURES_DIR / "model_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved model comparison bar chart.")


# =============================================================================
# Full Evaluation Pipeline
# =============================================================================

def evaluate_all_models(
    model_results: Dict[str, Dict[str, Any]],
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    X_test: pd.DataFrame, y_test: pd.Series,
) -> pd.DataFrame:
    """
    Run the complete evaluation pipeline for all trained models.

    Steps:
    1. Compute metrics on all splits for each model
    2. Generate confusion matrices
    3. Plot ROC and PR curves
    4. Create comparison table and bar chart

    Args:
        model_results: Output from model_training.train_all_models().
        X_train, y_train, X_val, y_val, X_test, y_test: Data splits.

    Returns:
        Comparison DataFrame.
    """
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    all_metrics = {}
    training_times = {}
    descriptions = {}

    for name, result in model_results.items():
        pipeline = result["pipeline"]

        # Compute metrics on all splits
        metrics = evaluate_model(
            pipeline,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            model_name=name,
        )
        all_metrics[name] = metrics
        training_times[name] = result["train_time"]
        descriptions[name] = result["description"]

        # Plot confusion matrices (test set)
        plot_confusion_matrix(pipeline, X_test, y_test, name, "Test")

    # Multi-model comparison plots
    plot_roc_curves(model_results, X_test, y_test)
    plot_precision_recall_curves(model_results, X_test, y_test)

    # Comparison table
    comparison_df = create_comparison_table(all_metrics, training_times, descriptions)
    plot_comparison_bar_chart(comparison_df)

    # Select best model
    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info(f"\n🏆 Best model: {best_model_name} (highest Validation ROC-AUC)")

    return comparison_df


# =============================================================================
# Decision Threshold Tuning
# =============================================================================

def tune_decision_threshold(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """
    Find the probability cutoff that maximizes F1 on the validation set.

    The default 0.5 cutoff is rarely optimal for imbalanced problems —
    with ~27% churners, a lower threshold typically trades a little
    precision for substantially better recall. The tuned value is saved
    with the model metadata and applied by the API at serving time.

    Args:
        pipeline: Fitted pipeline of the selected best model.
        X_val: Validation features (raw).
        y_val: Validation labels.

    Returns:
        Optimal threshold as a float in (0, 1).
    """
    probs = pipeline.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)

    # precision/recall have one more element than thresholds; align them
    precision, recall = precision[:-1], recall[:-1]
    denom = precision + recall
    f1_scores = np.divide(
        2 * precision * recall, denom,
        out=np.zeros_like(denom), where=denom > 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])

    logger.info(
        f"Tuned decision threshold on validation set: {best_threshold:.3f} "
        f"(F1={f1_scores[best_idx]:.4f}, "
        f"precision={precision[best_idx]:.4f}, recall={recall[best_idx]:.4f})"
    )
    return best_threshold
