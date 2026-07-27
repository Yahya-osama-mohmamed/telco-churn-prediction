import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    clean_data, build_preprocessing_pipeline, BinaryEncoder,
)
from src.feature_engineering import FeatureEngineer
from src.config import TARGET, ID_COLUMN, ENGINEERED_NUMERIC, ENGINEERED_FLAGS

@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        ID_COLUMN: ["C1", "C2", "C3", "C4"],
        "gender": ["Female", "Male", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "Yes", "No"],
        "Dependents": ["No", "No", "Yes", "Yes"],
        "tenure": [1, 34, 2, 45],
        "PhoneService": ["No", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes", "No"],
        "InternetService": ["DSL", "DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Electronic check", "Bank transfer"],
        "MonthlyCharges": [29.85, 56.95, 100.20, 20.00],
        "TotalCharges": ["29.85", " ", "200.40", "900.00"],
        TARGET: ["No", "No", "Yes", "No"]
    })

def test_clean_data(sample_raw_data):
    """Test the data cleaning step (types, missing values, duplicates)."""
    # Create a duplicate row (same customerID) to test duplicate removal
    df_with_dupes = pd.concat([sample_raw_data, sample_raw_data.iloc[[0]]], ignore_index=True)

    cleaned = clean_data(df_with_dupes)

    # ID column should be dropped
    assert ID_COLUMN not in cleaned.columns

    # Target should be encoded as 0/1
    assert list(cleaned[TARGET]) == [0, 0, 1, 0]

    # Duplicate customer IDs should be removed
    assert len(cleaned) == 4

    # TotalCharges should be converted to float
    assert cleaned["TotalCharges"].dtype == float

    # The blank string in TotalCharges (index 1) should be imputed with 0.0
    assert cleaned.loc[1, "TotalCharges"] == 0.0


def test_clean_data_keeps_distinct_customers_with_identical_profiles(sample_raw_data):
    """Two different customers with identical features must both survive cleaning."""
    twin = sample_raw_data.iloc[[0]].copy()
    twin[ID_COLUMN] = "C5"  # different customer, same profile
    df = pd.concat([sample_raw_data, twin], ignore_index=True)

    cleaned = clean_data(df)
    assert len(cleaned) == 5


def test_binary_encoder():
    """Test the custom BinaryEncoder for Yes/No and Male/Female columns."""
    df = pd.DataFrame({
        "Partner": ["Yes", "No", " Yes "],  # whitespace should be tolerated
        "gender": ["Male", "Female", "Male"],
        "Numeric": [1, 0, 1],  # non-object columns pass through unchanged
    })

    encoder = BinaryEncoder()
    encoded = encoder.fit_transform(df)

    assert list(encoded["Partner"]) == [1, 0, 1]
    assert list(encoded["gender"]) == [1, 0, 1]
    assert list(encoded["Numeric"]) == [1, 0, 1]


def test_binary_encoder_rejects_unknown_values():
    """Non-binary values in a supposedly binary column should fail fast."""
    df = pd.DataFrame({"Partner": ["Yes", "Maybe", "No"]})
    encoder = BinaryEncoder()
    with pytest.raises(ValueError, match="Partner"):
        encoder.fit_transform(df)


def test_binary_encoder_feature_names():
    """BinaryEncoder must support get_feature_names_out (one-to-one mapping)."""
    df = pd.DataFrame({"Partner": ["Yes", "No"], "gender": ["Male", "Female"]})
    encoder = BinaryEncoder().fit(df)
    assert list(encoder.get_feature_names_out()) == ["Partner", "gender"]
    assert list(encoder.get_feature_names_out(["a", "b"])) == ["a", "b"]


def test_preprocessing_pipeline_shapes(sample_raw_data):
    """Test the full scikit-learn ColumnTransformer pipeline."""
    # Clean first
    cleaned = clean_data(sample_raw_data)
    X = cleaned.drop(columns=[TARGET])

    # Define features based on the sample data columns
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    bin_features = ["gender", "Partner", "Dependents", "PhoneService"]
    cat_features = ["MultipleLines", "InternetService", "Contract", "PaymentMethod"]

    preprocessor = build_preprocessing_pipeline(
        numeric_features=num_features,
        binary_features=bin_features,
        multiclass_features=cat_features,
        flag_features=["SeniorCitizen"],
    )

    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)

    # All outputs should be numeric
    assert np.issubdtype(X_transformed.dtype, np.number)

    # Check scaling (mean should be approx 0)
    # The numeric features are the first 3 columns in the transformed output
    assert np.allclose(X_transformed[:, 0:3].mean(axis=0), 0, atol=1e-5)

    # get_feature_names_out must work (this was the original pipeline crash)
    names = list(preprocessor.get_feature_names_out())
    assert len(names) == X_transformed.shape[1]
    assert "bin__gender" in names
    assert "flag__SeniorCitizen" in names


def test_full_pipeline_single_row_inference(sample_raw_data):
    """
    Train/serve consistency: a pipeline fitted on training data must
    transform a single raw customer row (as the API receives it) without
    error, even if the row includes extra columns like customerID.
    """
    from sklearn.pipeline import Pipeline

    cleaned = clean_data(sample_raw_data)
    X = cleaned.drop(columns=[TARGET])

    preprocessor = build_preprocessing_pipeline(
        numeric_features=["tenure", "MonthlyCharges", "TotalCharges"] + ENGINEERED_NUMERIC,
        binary_features=["gender", "Partner", "Dependents", "PhoneService"],
        multiclass_features=["MultipleLines", "InternetService", "Contract",
                             "PaymentMethod", "tenure_group"],
        flag_features=["SeniorCitizen"] + ENGINEERED_FLAGS,
    )
    pipeline = Pipeline([
        ("features", FeatureEngineer()),
        ("preprocessor", preprocessor),
    ])
    pipeline.fit(X)

    # Single raw row with an extra column — must be ignored, not crash
    single = sample_raw_data.iloc[[2]].drop(columns=[TARGET]).copy()
    single["TotalCharges"] = pd.to_numeric(single["TotalCharges"])
    transformed = pipeline.transform(single)

    assert transformed.shape[0] == 1
    assert np.issubdtype(np.asarray(transformed).dtype, np.number)
    assert not np.isnan(np.asarray(transformed, dtype=float)).any()
