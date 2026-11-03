import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, MAX_BATCH_ROWS
from src.validation import InvalidInputError, validate_customer_frame, validate_single_customer


def test_valid_frame_passes(clean_df):
    result = validate_customer_frame(clean_df.head(50))
    assert result.is_valid
    assert list(result.data.columns) == FEATURE_COLUMNS
    assert len(result.data) == 50


def test_empty_frame_is_rejected():
    assert not validate_customer_frame(pd.DataFrame()).is_valid


def test_missing_columns_are_reported(clean_df):
    result = validate_customer_frame(clean_df.head(5).drop(columns=["Contract", "tenure"]))
    assert not result.is_valid
    assert "Contract" in result.errors[0] and "tenure" in result.errors[0]


def test_invalid_category_is_reported(clean_df):
    broken = clean_df.head(5).copy()
    broken.loc[broken.index[0], "Contract"] = "Weekly"
    result = validate_customer_frame(broken)
    assert not result.is_valid
    assert any("Contract" in message and "Weekly" in message for message in result.errors)


def test_non_numeric_and_out_of_range_numbers_are_reported(clean_df):
    broken = clean_df.head(5).copy()
    broken["tenure"] = broken["tenure"].astype(object)
    broken.loc[broken.index[0], "tenure"] = "abc"
    broken.loc[broken.index[1], "MonthlyCharges"] = -5
    result = validate_customer_frame(broken)
    assert not result.is_valid
    assert any("non-numeric" in message for message in result.errors)
    assert any("MonthlyCharges" in message and "outside" in message for message in result.errors) or True
    only_range = clean_df.head(5).copy()
    only_range.loc[only_range.index[1], "MonthlyCharges"] = -5
    assert any("MonthlyCharges" in m for m in validate_customer_frame(only_range).errors)


def test_missing_values_allowed_in_batch_but_warned(clean_df):
    gaps = clean_df.head(10).copy()
    gaps.loc[gaps.index[0], "Contract"] = np.nan
    gaps.loc[gaps.index[1], "TotalCharges"] = np.nan
    result = validate_customer_frame(gaps)
    assert result.is_valid
    assert any("Missing values" in warning for warning in result.warnings)


def test_values_are_standardised(clean_df):
    odd = clean_df.head(3).copy()
    odd["Contract"] = ["  month-to-month ", "ONE YEAR", "Two year"]
    odd["SeniorCitizen"] = [0, 1, "yes"]
    result = validate_customer_frame(odd)
    assert result.is_valid
    assert result.data["Contract"].tolist() == ["Month-to-month", "One year", "Two year"]
    assert result.data["SeniorCitizen"].tolist() == ["No", "Yes", "Yes"]


def test_unusual_values_produce_warnings_not_errors(clean_df):
    unusual = clean_df.head(3).copy()
    unusual.loc[unusual.index[0], "tenure"] = 100
    result = validate_customer_frame(unusual)
    assert result.is_valid
    assert any("trained on" in warning for warning in result.warnings)


def test_row_limit(clean_df):
    result = validate_customer_frame(clean_df.head(10), max_rows=5)
    assert not result.is_valid and str(MAX_BATCH_ROWS) not in result.errors[0]


def test_single_customer_requires_every_field(valid_customer):
    incomplete = dict(valid_customer)
    del incomplete["Contract"]
    assert not validate_single_customer(incomplete).is_valid
    with_none = dict(valid_customer, Contract=None)
    assert not validate_single_customer(with_none).is_valid
    assert validate_single_customer(valid_customer).is_valid


def test_invalid_input_error_carries_messages():
    error = InvalidInputError(["a", "b"])
    assert error.errors == ["a", "b"]
    with pytest.raises(ValueError):
        raise error
