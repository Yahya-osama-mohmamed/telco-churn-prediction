"""
Model Explainability — SHAP-based interpretation.

Uses SHAP (SHapley Additive exPlanations) to provide:
- Global feature importance (which features matter most overall)
- Local explanations (why the model made a specific prediction)
- Feature interaction effects

SHAP values are grounded in game theory and provide theoretically
consistent, locally accurate feature attributions.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any

from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def get_shap_explainer(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    model_name: str = "Model",
) -> shap.Explainer:
    """
    Create a SHAP explainer appropriate for the model type.

    Uses TreeExplainer for tree-based models (XGBoost, LightGBM)
    and KernelExplainer as a fallback for other model types.

    Args:
        pipeline: Fitted sklearn Pipeline (preprocessor + classifier).
        X_train: Training features (raw, before preprocessing).
        model_name: Model name for logging.

    Returns:
        SHAP Explainer instance.
    """
    # Extract the classifier from the pipeline
    classifier = pipeline.named_steps["classifier"]

    # Transform training data through every step except the classifier
    X_train_df = transform_features(pipeline, X_train)

    # Choose appropriate explainer
    model_type = type(classifier).__name__
    if model_type in ("XGBClassifier", "LGBMClassifier", "RandomForestClassifier"):
        logger.info(f"Using TreeExplainer for {model_name} ({model_type})")
        explainer = shap.TreeExplainer(classifier)
    elif hasattr(classifier, "coef_"):
        # Exact and fast for linear models (explains the log-odds margin).
        # KernelExplainer here would need ~2k model evals per explained row.
        logger.info(f"Using LinearExplainer for {model_name} ({model_type})")
        explainer = shap.LinearExplainer(classifier, X_train_df)
    else:
        logger.info(f"Using KernelExplainer for {model_name} ({model_type})")
        # Use a sample of training data for KernelExplainer (it's slow)
        background = shap.sample(X_train_df, min(100, len(X_train_df)))
        explainer = shap.KernelExplainer(classifier.predict_proba, background)

    return explainer


def transform_features(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Run raw features through every pipeline step except the final classifier
    and return the result as a DataFrame with proper feature names.

    Args:
        pipeline: Fitted Pipeline (feature engineering + preprocessing + classifier).
        X: Raw input features.

    Returns:
        Transformed DataFrame in model-input space.
    """
    transformer = Pipeline(pipeline.steps[:-1])
    X_transformed = transformer.transform(X)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        logger.warning("Could not extract feature names from preprocessor.")
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    return pd.DataFrame(np.asarray(X_transformed), columns=feature_names)


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    explainer: shap.Explainer,
    max_samples: int = 500,
) -> tuple:
    """
    Compute SHAP values for a dataset.

    Args:
        pipeline: Fitted Pipeline.
        X: Raw features (before preprocessing).
        explainer: SHAP Explainer instance.
        max_samples: Maximum number of samples to explain (for speed).

    Returns:
        Tuple of (shap_values, X_transformed as DataFrame).
    """
    # Sample if dataset is large
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X

    # Transform through feature engineering + preprocessing
    X_df = transform_features(pipeline, X_sample)

    # Compute SHAP values
    logger.info(f"Computing SHAP values for {len(X_df)} samples...")
    shap_values = explainer.shap_values(X_df)

    # For binary classification, explainers may return a list [class_0, class_1]
    # or a 3D array (n_samples, n_features, n_classes) — take the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values, X_df


def plot_shap_summary(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    model_name: str = "Model",
) -> None:
    """
    Create a SHAP beeswarm summary plot.

    Each point represents a single feature value for a single prediction.
    The x-axis shows the SHAP value (impact on prediction), and the
    color shows the feature value (red = high, blue = low).

    This is the most informative single visualization of model behavior.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_df,
        plot_type="dot",
        show=False,
        max_display=15,
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_summary.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    logger.info("Saved SHAP summary plot (beeswarm).")


def plot_shap_bar(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    model_name: str = "Model",
) -> None:
    """
    Create a SHAP bar plot showing mean absolute SHAP values.

    This shows which features have the largest impact on predictions
    on average, regardless of direction.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_df,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.title(f"Global Feature Importance (SHAP) — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "shap_bar.png", dpi=150, bbox_inches="tight"
    )
    plt.close("all")
    logger.info("Saved SHAP bar plot (global importance).")


def plot_shap_waterfall(
    explainer: shap.Explainer,
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    sample_idx: int = 0,
    model_name: str = "Model",
) -> None:
    """
    Create a SHAP waterfall plot for a single prediction.

    Shows how each feature pushes the prediction from the base value
    (average prediction) to the final output. This is the best way
    to explain an individual prediction to stakeholders.
    """
    try:
        # Create an Explanation object
        if hasattr(explainer, "expected_value"):
            expected = explainer.expected_value
            if isinstance(expected, (list, np.ndarray)):
                expected = expected[1] if len(expected) > 1 else expected[0]
        else:
            expected = 0.0

        explanation = shap.Explanation(
            values=shap_values[sample_idx],
            base_values=expected,
            data=X_df.iloc[sample_idx].values,
            feature_names=X_df.columns.tolist(),
        )

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False, max_display=12)
        plt.title(
            f"SHAP Waterfall — Single Prediction ({model_name})",
            fontsize=13, fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "shap_waterfall.png", dpi=150, bbox_inches="tight"
        )
        plt.close("all")
        logger.info("Saved SHAP waterfall plot.")
    except Exception as e:
        logger.warning(f"Could not create waterfall plot: {e}")


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_df: pd.DataFrame,
    top_n: int = 3,
    model_name: str = "Model",
) -> None:
    """
    Create SHAP dependence plots for the top N most important features.

    Dependence plots show how the SHAP value of a feature changes as
    its value changes. The color shows the most interacting feature.
    """
    # Find top features by mean absolute SHAP value
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[-top_n:][::-1]
    top_features = [X_df.columns[i] for i in top_indices]

    for feature in top_features:
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                feature, shap_values, X_df,
                show=False,
                ax=ax,
            )
            ax.set_title(
                f"SHAP Dependence — {feature} ({model_name})",
                fontsize=13, fontweight="bold",
            )
            plt.tight_layout()
            safe_name = feature.replace(" ", "_").replace("/", "_")
            fig.savefig(
                FIGURES_DIR / f"shap_dependence_{safe_name}.png",
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not create dependence plot for {feature}: {e}")

    logger.info(f"Saved SHAP dependence plots for top {top_n} features.")


def run_explainability(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    model_name: str = "Best Model",
) -> Dict[str, Any]:
    """
    Run the full SHAP explainability pipeline.

    Steps:
    1. Create SHAP explainer
    2. Compute SHAP values on test set
    3. Generate summary, bar, waterfall, and dependence plots

    Args:
        pipeline: Fitted best model Pipeline.
        X_train: Training features (for explainer background).
        X_test: Test features (to explain).
        model_name: Model name for plot titles.

    Returns:
        Dictionary with SHAP values and feature importance ranking.
    """
    logger.info("=" * 60)
    logger.info(f"SHAP EXPLAINABILITY — {model_name}")
    logger.info("=" * 60)

    # Create explainer
    explainer = get_shap_explainer(pipeline, X_train, model_name)

    # Compute SHAP values
    shap_values, X_df = compute_shap_values(pipeline, X_test, explainer)

    # Generate all plots
    plot_shap_summary(shap_values, X_df, model_name)
    plot_shap_bar(shap_values, X_df, model_name)
    plot_shap_waterfall(explainer, shap_values, X_df, sample_idx=0, model_name=model_name)
    plot_shap_dependence(shap_values, X_df, top_n=3, model_name=model_name)

    # Feature importance ranking
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info("\nTop 10 features by SHAP importance:")
    for _, row in importance_df.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

    return {
        "shap_values": shap_values,
        "feature_importance": importance_df,
        "explainer": explainer,
    }
