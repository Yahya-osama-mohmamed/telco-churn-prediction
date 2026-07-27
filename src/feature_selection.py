"""
Feature Selection — Identify the most predictive features.

Uses three complementary methods:
1. Correlation filtering — remove highly correlated feature pairs
2. Mutual Information — rank features by information gain with target
3. Model-based importance — Random Forest feature importances

The final feature set is the union of top features from methods 2 & 3,
after applying the correlation filter from method 1.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict

from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

from src.config import RANDOM_STATE, TARGET, FIGURES_DIR
from src.logger import get_logger

logger = get_logger(__name__)


def correlation_filter(
    df: pd.DataFrame,
    threshold: float = 0.90,
    target: str = TARGET,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove one feature from each pair of highly correlated features.

    When two features have |correlation| > threshold, the one with
    lower correlation to the target variable is dropped.

    Args:
        df: DataFrame with numeric features.
        threshold: Absolute correlation cutoff (default 0.90).
        target: Target column name.

    Returns:
        Tuple of (filtered DataFrame, list of dropped column names).
    """
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=[np.number])

    if target in numeric_df.columns:
        numeric_for_corr = numeric_df.drop(columns=[target])
        target_corr = numeric_df.corrwith(numeric_df[target]).abs()
    else:
        numeric_for_corr = numeric_df
        target_corr = pd.Series(0, index=numeric_for_corr.columns)

    corr_matrix = numeric_for_corr.corr().abs()

    # Find pairs above threshold (upper triangle only to avoid double-counting)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    dropped_cols = []
    for col in upper_tri.columns:
        high_corr_cols = upper_tri.index[upper_tri[col] > threshold].tolist()
        for paired_col in high_corr_cols:
            # Drop the feature with lower target correlation
            if paired_col not in dropped_cols and col not in dropped_cols:
                if target_corr.get(col, 0) >= target_corr.get(paired_col, 0):
                    dropped_cols.append(paired_col)
                else:
                    dropped_cols.append(col)
                logger.info(
                    f"Correlation filter: dropping '{dropped_cols[-1]}' "
                    f"(r={corr_matrix.loc[col, paired_col]:.3f} with '{col if dropped_cols[-1] != col else paired_col}')"
                )

    df_filtered = df.drop(columns=[c for c in dropped_cols if c in df.columns])
    logger.info(
        f"Correlation filter: dropped {len(dropped_cols)} features. "
        f"Remaining: {df_filtered.shape[1]} columns."
    )
    return df_filtered, dropped_cols


def mutual_information_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Rank features by Mutual Information with the target.

    Mutual Information measures the amount of information obtained about
    the target from observing each feature. It captures non-linear
    relationships that Pearson correlation misses.

    Args:
        X: Feature DataFrame (numeric only).
        y: Target Series.
        top_n: Number of top features to return.

    Returns:
        DataFrame with features ranked by MI score.
    """
    # Select only numeric features for MI calculation
    X_numeric = X.select_dtypes(include=[np.number])

    mi_scores = mutual_info_classif(
        X_numeric, y,
        random_state=RANDOM_STATE,
        n_neighbors=5,
    )

    mi_df = pd.DataFrame({
        "feature": X_numeric.columns,
        "mi_score": mi_scores,
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    logger.info(f"Top {top_n} features by Mutual Information:")
    for _, row in mi_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['mi_score']:.4f}")

    # Save visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=mi_df.head(top_n),
        x="mi_score", y="feature",
        hue="feature",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Feature Ranking — Mutual Information", fontsize=14, fontweight="bold")
    ax.set_xlabel("Mutual Information Score")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_selection_mi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return mi_df


def model_based_importance(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Rank features using Random Forest feature importances.

    Random Forest importance measures how much each feature contributes
    to reducing impurity (Gini) across all trees. This is a good
    complement to MI because it accounts for feature interactions.

    Args:
        X: Feature DataFrame (numeric only).
        y: Target Series.
        top_n: Number of top features to return.

    Returns:
        DataFrame with features ranked by importance.
    """
    X_numeric = X.select_dtypes(include=[np.number])

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_numeric, y)

    importance_df = pd.DataFrame({
        "feature": X_numeric.columns,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    logger.info(f"Top {top_n} features by Random Forest importance:")
    for _, row in importance_df.head(top_n).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Save visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=importance_df.head(top_n),
        x="importance", y="feature",
        hue="feature",
        palette="magma",
        legend=False,
        ax=ax,
    )
    ax.set_title("Feature Ranking — Random Forest Importance", fontsize=14, fontweight="bold")
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_selection_rf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return importance_df


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int = 15,
    corr_threshold: float = 0.90,
) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
    """
    Run the full feature selection pipeline.

    Strategy:
    1. Apply correlation filter to remove redundant features
    2. Rank remaining features with MI and RF importance
    3. Take the union of top features from both methods

    Args:
        X: Feature DataFrame.
        y: Target Series.
        top_n: Number of top features from each method.
        corr_threshold: Correlation threshold for filtering.

    Returns:
        Tuple of (selected feature names, dict of ranking DataFrames).
    """
    logger.info("=" * 60)
    logger.info("FEATURE SELECTION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Correlation filter
    X_filtered, dropped = correlation_filter(
        X.assign(**{TARGET: y}), threshold=corr_threshold, target=TARGET,
    )
    if TARGET in X_filtered.columns:
        X_filtered = X_filtered.drop(columns=[TARGET])

    # Step 2: Mutual Information ranking
    mi_df = mutual_information_ranking(X_filtered, y, top_n=top_n)

    # Step 3: Model-based importance ranking
    rf_df = model_based_importance(X_filtered, y, top_n=top_n)

    # Step 4: Union of top features from both methods
    mi_top = set(mi_df.head(top_n)["feature"].tolist())
    rf_top = set(rf_df.head(top_n)["feature"].tolist())
    selected = sorted(mi_top | rf_top)

    # Only keep features that exist in X_filtered
    selected = [f for f in selected if f in X_filtered.columns]

    logger.info(f"Selected {len(selected)} features (union of MI + RF top-{top_n}):")
    for feat in selected:
        logger.info(f"  ✓ {feat}")

    rankings = {"mutual_information": mi_df, "random_forest": rf_df}
    return selected, rankings
