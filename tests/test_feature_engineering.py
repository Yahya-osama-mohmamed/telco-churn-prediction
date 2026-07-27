import pytest
import pandas as pd
from src.feature_engineering import (
    create_tenure_group,
    create_avg_monthly_charge,
    create_service_count,
    create_has_security_backup,
    create_high_value_short_tenure,
    FeatureEngineer,
)

@pytest.fixture
def sample_feature_data():
    return pd.DataFrame({
        "tenure": [0, 5, 24, 70],
        "MonthlyCharges": [50.0, 100.0, 20.0, 110.0],
        "TotalCharges": [0.0, 500.0, 480.0, 7700.0],
        "PhoneService": ["Yes", "No", "Yes", "Yes"],
        "InternetService": ["No", "DSL", "Fiber optic", "Fiber optic"],
        "OnlineSecurity": ["No internet service", "Yes", "No", "Yes"],
        "OnlineBackup": ["No internet service", "Yes", "Yes", "Yes"]
    })

def test_create_tenure_group(sample_feature_data):
    """Test tenure binning logic."""
    result = create_tenure_group(sample_feature_data)
    assert "tenure_group" in result.columns
    # 0 -> 0-12, 5 -> 0-12, 24 -> 13-24, 70 -> 61+
    assert list(result["tenure_group"]) == ["0-12", "0-12", "13-24", "61+"]

def test_create_tenure_group_open_ended(sample_feature_data):
    """Tenure beyond 72 months must fall in the last bin, not become NaN."""
    df = sample_feature_data.copy()
    df.loc[0, "tenure"] = 90
    result = create_tenure_group(df)
    assert result.loc[0, "tenure_group"] == "61+"
    assert not result["tenure_group"].isna().any()

def test_create_avg_monthly_charge(sample_feature_data):
    """Test average monthly charge calculation, especially for tenure=0."""
    result = create_avg_monthly_charge(sample_feature_data)

    # For tenure=0, denominator should be clipped to 1
    assert result.loc[0, "avg_monthly_charge"] == 0.0

    # For normal rows: 500/5, 480/24, 7700/70
    assert list(result["avg_monthly_charge"][1:]) == [100.0, 20.0, 110.0]

def test_create_service_count(sample_feature_data):
    """A service is active unless its value is a 'No ...' variant."""
    result = create_service_count(sample_feature_data)

    # Row 0: PhoneService (1) — InternetService="No"
    # Row 1: DSL, OnlineSecurity, OnlineBackup (3)
    # Row 2: PhoneService, Fiber optic, OnlineBackup (3)
    # Row 3: PhoneService, Fiber optic, OnlineSecurity, OnlineBackup (4)
    assert list(result["service_count"]) == [1, 3, 3, 4]

def test_create_has_security_backup(sample_feature_data):
    """Test the combined flag for both security AND backup."""
    result = create_has_security_backup(sample_feature_data)

    # Only Row 1 and Row 3 have BOTH set to "Yes"
    assert list(result["has_security_backup"]) == [0, 1, 0, 1]

def test_create_high_value_short_tenure(sample_feature_data):
    """Test the risk indicator for high spending, short tenure customers."""
    result = create_high_value_short_tenure(sample_feature_data)

    # Median MonthlyCharges is 75.0
    # Row 0: MonthlyCharges=50 (<75), tenure=0 (<12) -> 0
    # Row 1: MonthlyCharges=100 (>75), tenure=5 (<12) -> 1
    # Row 2: MonthlyCharges=20 (<75), tenure=24 (>12) -> 0
    # Row 3: MonthlyCharges=110 (>75), tenure=70 (>12) -> 0
    assert list(result["high_value_short_tenure"]) == [0, 1, 0, 0]

def test_create_high_value_short_tenure_fixed_threshold(sample_feature_data):
    """An explicit threshold must override the per-DataFrame median."""
    result = create_high_value_short_tenure(sample_feature_data, charge_threshold=40.0)
    # Rows 0 and 1 have tenure < 12; both charge above 40
    assert list(result["high_value_short_tenure"]) == [1, 1, 0, 0]


def test_feature_engineer_learns_median_at_fit(sample_feature_data):
    """
    Train/serve consistency: the MonthlyCharges threshold must come from
    the TRAINING data seen in fit, not from the data being transformed.
    A single-row transform would otherwise always produce 0 (its own
    median can never be exceeded).
    """
    fe = FeatureEngineer()
    fe.fit(sample_feature_data)
    assert fe.monthly_charges_median_ == 75.0

    # Single new customer: charges 100 > 75 (train median), tenure 3 < 12
    single = pd.DataFrame({
        "tenure": [3],
        "MonthlyCharges": [100.0],
        "TotalCharges": [300.0],
        "PhoneService": ["Yes"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["No"],
    })
    result = fe.transform(single)
    assert result.loc[0, "high_value_short_tenure"] == 1

    engineered_cols = {
        "tenure_group", "avg_monthly_charge", "service_count",
        "has_security_backup", "high_value_short_tenure", "charge_tenure_ratio",
    }
    assert engineered_cols.issubset(result.columns)
