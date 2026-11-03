import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, RANDOM_STATE
from src.preprocessing import (
    build_preprocessor, encode_target, make_train_test_split, map_encoded_to_original, normalize_senior_citizen,
)


def test_split_is_stratified_and_reproducible(clean_df):
    X_train, X_test, y_train, y_test = make_train_test_split(clean_df)
    assert len(X_train) + len(X_test) == len(clean_df)
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert set(X_train.index).isdisjoint(X_test.index)
    again = make_train_test_split(clean_df)
    pd.testing.assert_frame_equal(X_train, again[0])


def test_features_exclude_id_and_target(clean_df):
    X_train, _, _, _ = make_train_test_split(clean_df)
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert "Churn" not in X_train.columns and "customerID" not in X_train.columns


def test_preprocessor_is_fitted_on_training_data_only(clean_df):
    X_train, X_test, _, _ = make_train_test_split(clean_df)
    preprocessor = build_preprocessor().fit(X_train)
    scaler = preprocessor.named_transformers_["num"].named_steps["scaler"]
    assert scaler.mean_[0] == pytest.approx(X_train["tenure"].mean())
    assert scaler.mean_[0] != pytest.approx(clean_df["tenure"].mean())  # no leakage from test rows


def test_preprocessor_output_has_no_missing_values(clean_df):
    X_train, X_test, _, _ = make_train_test_split(clean_df)
    preprocessor = build_preprocessor().fit(X_train)
    with_gaps = X_test.head(20).copy()
    with_gaps.loc[with_gaps.index[0], "tenure"] = np.nan
    with_gaps.loc[with_gaps.index[1], "Contract"] = np.nan
    output = preprocessor.transform(with_gaps)
    assert not np.isnan(output).any()
    assert output.shape[0] == 20


def test_unknown_category_does_not_crash(clean_df):
    X_train, X_test, _, _ = make_train_test_split(clean_df)
    preprocessor = build_preprocessor().fit(X_train)
    row = X_test.head(1).copy()
    row["Contract"] = "Five year"
    assert preprocessor.transform(row).shape[0] == 1


def test_encode_target():
    assert encode_target(pd.Series(["Yes", "No", "Yes"])).tolist() == [1, 0, 1]
    with pytest.raises(ValueError):
        encode_target(pd.Series(["Yes", "Maybe"]))


def test_normalize_senior_citizen():
    result = normalize_senior_citizen(pd.Series([0, 1, "Yes", "no", "x", None], dtype=object))
    assert result.iloc[:4].tolist() == ["No", "Yes", "Yes", "No"]
    assert result.iloc[4:].isna().all()


def test_encoded_names_map_back_to_original_columns(clean_df):
    X_train, _, _, _ = make_train_test_split(clean_df)
    names = build_preprocessor().fit(X_train).get_feature_names_out()
    originals = map_encoded_to_original(names)
    assert set(originals) == set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)
