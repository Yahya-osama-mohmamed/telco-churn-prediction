"""Tests for the transformers the notebook trains and the API serves.

These cover the two failure modes that would be invisible in a notebook run but
fatal in production: a feature that is computed differently at serving time than
during training, and a transformer that silently accepts bad input.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline_lib import BinaryEncoder, FeatureEngineer, coerce_numeric_columns, engineer_features


def make_rows(n: int = 6) -> pd.DataFrame:
    """A small frame with the columns the pipeline actually reads."""
    return pd.DataFrame({
        "gender": ["Female", "Male"] * (n // 2),
        "SeniorCitizen": [0, 1] * (n // 2),
        "Partner": ["Yes", "No"] * (n // 2),
        "Dependents": ["No", "Yes"] * (n // 2),
        "tenure": [1, 12, 25, 40, 55, 70][:n],
        "PhoneService": ["Yes"] * n,
        "MultipleLines": ["No", "Yes", "No phone service", "Yes", "No", "Yes"][:n],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic", "No"][:n],
        "OnlineSecurity": ["Yes", "No", "No internet service", "Yes", "No", "Yes"][:n],
        "OnlineBackup": ["Yes", "No", "No internet service", "No", "Yes", "Yes"][:n],
        "DeviceProtection": ["No"] * n,
        "TechSupport": ["No"] * n,
        "StreamingTV": ["No"] * n,
        "StreamingMovies": ["No"] * n,
        "Contract": ["Month-to-month", "One year", "Two year"] * (n // 3),
        "PaperlessBilling": ["Yes", "No"] * (n // 2),
        "PaymentMethod": ["Electronic check", "Mailed check"] * (n // 2),
        "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70, 99.65][:n],
        "TotalCharges": [29.85, 1889.5, 108.15, 1840.75, 151.65, 820.5][:n],
    })


class TestFeatureEngineer:
    def test_adds_every_expected_feature(self):
        out = FeatureEngineer().fit(make_rows()).transform(make_rows())
        for col in ("tenure_group", "avg_monthly_charge", "service_count",
                    "has_security_backup", "high_value_short_tenure", "charge_tenure_ratio"):
            assert col in out.columns

    def test_threshold_comes_from_fit_not_transform(self):
        """The high-value flag must use the TRAINING median, not the median of
        whatever is being scored - otherwise a single-row API request would
        compare a customer against themselves."""
        train = make_rows()
        fe = FeatureEngineer().fit(train)
        learned = fe.monthly_charges_median_

        # One expensive customer, scored alone. Their own median is their own
        # charge, so a transform-time median would make the flag always 0.
        single = train.iloc[[5]].copy()
        out = fe.transform(single)
        expected = int(single["MonthlyCharges"].iloc[0] > learned and single["tenure"].iloc[0] < 12)
        assert out["high_value_short_tenure"].iloc[0] == expected

    def test_service_count_includes_internet_service(self):
        """InternetService reads DSL / Fiber optic / No - never 'Yes'. An
        equals-Yes check would silently drop it from the count."""
        df = make_rows()
        out = FeatureEngineer().fit(df).transform(df)
        dsl = out[out["InternetService"] == "DSL"]["service_count"]
        none = out[out["InternetService"] == "No"]["service_count"]
        assert dsl.min() > none.min()

    def test_tenure_group_is_open_ended(self):
        """Tenure beyond the training range must not produce NaN."""
        df = make_rows()
        fe = FeatureEngineer().fit(df)
        future = df.iloc[[0]].copy()
        future["tenure"] = 999
        out = fe.transform(future)
        assert out["tenure_group"].iloc[0] == "61+"
        assert out["tenure_group"].notna().all()

    def test_missing_optional_column_does_not_produce_nan(self):
        """A dropped column should yield False for the whole frame, index-aligned."""
        df = make_rows().drop(columns=["OnlineSecurity"])
        out = FeatureEngineer().fit(df).transform(df)
        assert out["has_security_backup"].notna().all()
        assert (out["has_security_backup"] == 0).all()

    def test_string_totalcharges_is_coerced(self):
        """Raw CSV uploads deliver blanks for never-billed customers."""
        df = make_rows()
        df["TotalCharges"] = df["TotalCharges"].astype(str)
        df.loc[0, "TotalCharges"] = " "
        out = FeatureEngineer().fit(df).transform(df)
        assert out["avg_monthly_charge"].notna().all()
        assert out["avg_monthly_charge"].iloc[0] == 0.0

    def test_requires_monthly_charges(self):
        with pytest.raises(ValueError, match="MonthlyCharges"):
            FeatureEngineer().fit(make_rows().drop(columns=["MonthlyCharges"]))

    def test_feature_names_out_matches_transform(self):
        df = make_rows()
        fe = FeatureEngineer().fit(df)
        assert list(fe.get_feature_names_out()) == list(fe.transform(df).columns)


class TestBinaryEncoder:
    def test_maps_to_one_and_zero(self):
        df = pd.DataFrame({"Partner": ["Yes", "No"], "gender": ["Male", "Female"]})
        out = BinaryEncoder().fit(df).transform(df)
        assert out["Partner"].tolist() == [1, 0]
        assert out["gender"].tolist() == [1, 0]

    def test_tolerates_surrounding_whitespace(self):
        df = pd.DataFrame({"Partner": [" Yes", "No "]})
        assert BinaryEncoder().fit(df).transform(df)["Partner"].tolist() == [1, 0]

    def test_rejects_unknown_values(self):
        """Failing loudly beats scoring a customer on garbage input."""
        df = pd.DataFrame({"Partner": ["Yes", "Maybe"]})
        with pytest.raises(ValueError, match="not binary"):
            BinaryEncoder().fit(df).transform(df)

    def test_names_are_one_to_one(self):
        df = pd.DataFrame({"Partner": ["Yes"], "gender": ["Male"]})
        enc = BinaryEncoder().fit(df)
        assert list(enc.get_feature_names_out()) == ["Partner", "gender"]


class TestEngineerFeaturesHelper:
    def test_explicit_threshold_is_respected(self):
        df = make_rows()
        low = engineer_features(df, charge_threshold=0.0)
        high = engineer_features(df, charge_threshold=1e9)
        assert low["high_value_short_tenure"].sum() >= high["high_value_short_tenure"].sum()
        assert high["high_value_short_tenure"].sum() == 0

    def test_coerce_handles_blank_totalcharges(self):
        df = pd.DataFrame({"tenure": [0], "MonthlyCharges": ["29.85"], "TotalCharges": [" "]})
        out = coerce_numeric_columns(df)
        assert out["TotalCharges"].iloc[0] == 0.0
        assert np.issubdtype(out["MonthlyCharges"].dtype, np.number)
