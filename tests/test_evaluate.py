import numpy as np
import pytest

from src.evaluate import (
    compute_classification_metrics, compute_confusion_matrix, compute_roc_curve, evaluate_pipeline,
    get_xgboost_feature_importance,
)
from src.preprocessing import make_train_test_split


def test_metrics_on_known_values():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.4, 0.9])
    metrics = compute_classification_metrics(y_true, y_proba)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_confusion_matrix_counts():
    matrix = compute_confusion_matrix(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.4, 0.9]))
    assert matrix == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_roc_curve_starts_at_origin_and_ends_at_one():
    curve = compute_roc_curve(np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.4, 0.6]))
    assert (curve["fpr"][0], curve["tpr"][0]) == (0.0, 0.0)
    assert (curve["fpr"][-1], curve["tpr"][-1]) == (1.0, 1.0)


def test_evaluate_pipeline_returns_all_parts(small_pipeline, sample_df):
    _, X_test, _, y_test = make_train_test_split(sample_df)
    result = evaluate_pipeline(small_pipeline, X_test, y_test)
    assert set(result["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in result["metrics"].values())
    assert sum(result["confusion_matrix"].values()) == len(y_test)
    assert "Churned" in result["classification_report"]


def test_feature_importance_is_per_original_feature(small_pipeline):
    table = get_xgboost_feature_importance(small_pipeline)
    assert table["feature"].is_unique
    assert table["importance"].sum() == pytest.approx(1.0, abs=1e-4)
