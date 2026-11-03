import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORY_OPTIONS, DATA_DICTIONARY, FEATURE_COLUMNS, TARGET_COLUMN
from src.data_loader import (
    clean_dataset, data_quality_report, find_unexpected_categories, generate_synthetic_dataset,
    load_raw_dataset,
)


def test_raw_dataset_loads_with_expected_columns():
    df = load_raw_dataset()
    assert len(df) > 1000
    assert {"customerID", "tenure", "MonthlyCharges", "TotalCharges", "Contract", "Churn"} <= set(df.columns)


def test_load_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_dataset(tmp_path / "nope.csv")


def test_load_raises_when_required_columns_missing(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_raw_dataset(bad)


def test_cleaning_fixes_total_charges_and_senior_citizen():
    raw = load_raw_dataset()
    assert raw["TotalCharges"].dtype != float  # blanks make the raw column text
    clean = clean_dataset(raw)
    assert pd.api.types.is_float_dtype(clean["TotalCharges"])
    assert clean["TotalCharges"].isna().sum() == 0
    assert set(clean["SeniorCitizen"].unique()) <= {"Yes", "No"}
    new_customers = clean[clean["tenure"] == 0]
    assert (new_customers["TotalCharges"] == 0).all()


def test_cleaning_does_not_modify_input_or_drop_rows():
    raw = load_raw_dataset()
    before = raw.copy()
    clean = clean_dataset(raw)
    pd.testing.assert_frame_equal(raw, before)
    assert len(clean) == len(raw)


def test_quality_report_contents(clean_df):
    report = data_quality_report(clean_df)
    assert report["rows"] == len(clean_df)
    assert report["duplicate_rows"] == 0
    assert 0 < report["churn_rate"] < 1
    assert set(report["churn_counts"]) == {"Yes", "No"}


def test_dataset_categories_match_config(clean_df):
    """The form options, validation and data must agree exactly."""
    assert find_unexpected_categories(clean_df) == {}
    for column in FEATURE_COLUMNS:
        assert column in clean_df.columns
        assert column in DATA_DICTIONARY


def test_synthetic_dataset_is_reproducible_and_compatible():
    first = generate_synthetic_dataset(500, seed=1)
    second = generate_synthetic_dataset(500, seed=1)
    pd.testing.assert_frame_equal(first, second)
    cleaned = clean_dataset(first)
    assert set(FEATURE_COLUMNS + [TARGET_COLUMN]) <= set(cleaned.columns)
    assert find_unexpected_categories(cleaned) == {}
    assert first["customerID"].str.startswith("SYN-").all()
    assert not np.isnan(first["MonthlyCharges"]).any()
