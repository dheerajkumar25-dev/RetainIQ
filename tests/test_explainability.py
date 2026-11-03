import numpy as np
import pytest

from src.config import FEATURE_COLUMNS
from src.explainability import (
    aggregate_by_original_feature, compute_shap_values, explain_single_customer, global_importance,
)
from src.recommendations import suggest_retention_actions
from src.preprocessing import make_train_test_split


def test_shap_values_add_up_to_model_output(small_pipeline, sample_df):
    _, X_test, _, _ = make_train_test_split(sample_df)
    values, base_value, encoded = compute_shap_values(small_pipeline, X_test.head(25))
    margin = small_pipeline.named_steps["model"].predict(encoded, output_margin=True)
    np.testing.assert_allclose(values.sum(axis=1) + base_value, margin, atol=1e-3)


def test_aggregation_preserves_totals(small_pipeline, sample_df):
    _, X_test, _, _ = make_train_test_split(sample_df)
    values, _, encoded = compute_shap_values(small_pipeline, X_test.head(10))
    grouped = aggregate_by_original_feature(values, encoded.columns)
    assert set(grouped.columns) == set(FEATURE_COLUMNS)
    np.testing.assert_allclose(grouped.sum(axis=1), values.sum(axis=1), atol=1e-4)


def test_single_customer_explanation(small_pipeline, sample_df):
    _, X_test, _, _ = make_train_test_split(sample_df)
    table = explain_single_customer(small_pipeline, X_test.head(1), top_n=5)
    assert list(table.columns) == ["feature", "value", "shap_value"]
    assert len(table) == 5
    assert table["shap_value"].abs().is_monotonic_decreasing


def test_global_importance_sorted(small_pipeline, sample_df):
    _, X_test, _, _ = make_train_test_split(sample_df)
    table = global_importance(small_pipeline, X_test.head(100))
    assert table["mean_abs_shap"].is_monotonic_decreasing
    assert (table["mean_abs_shap"] >= 0).all()


def test_retention_actions_follow_risk_drivers(small_pipeline, valid_customer):
    import pandas as pd
    customer = dict(valid_customer, Contract="Month-to-month", tenure=3)
    drivers = pd.DataFrame({"feature": ["Contract", "tenure", "TechSupport"],
                            "value": ["Month-to-month", 3, "Yes"], "shap_value": [0.8, 0.5, -0.2]})
    actions = suggest_retention_actions(customer, drivers)
    assert len(actions) == 2  # TechSupport is protective, so no rule fires for it
    assert suggest_retention_actions(customer, drivers.assign(shap_value=-1.0)) == []
