import json

import joblib
import numpy as np
import pytest

from src.config import MODEL_PATH, METRICS_PATH
from src.preprocessing import make_train_test_split
from src.train import build_pipelines, run_training


def test_all_three_models_are_defined():
    assert list(build_pipelines(2.7)) == ["Logistic Regression", "Random Forest", "XGBoost"]


def test_pipelines_train_and_predict_probabilities(sample_df):
    X_train, X_test, y_train, _ = make_train_test_split(sample_df)
    for name, pipeline in build_pipelines(2.7).items():
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        assert probabilities.shape == (len(X_test),), name
        assert ((probabilities >= 0) & (probabilities <= 1)).all(), name


def test_training_is_reproducible(sample_df):
    X_train, X_test, y_train, _ = make_train_test_split(sample_df)
    first = build_pipelines(2.7)["XGBoost"].fit(X_train, y_train).predict_proba(X_test)
    second = build_pipelines(2.7)["XGBoost"].fit(X_train, y_train).predict_proba(X_test)
    np.testing.assert_allclose(first, second)


def test_full_training_run_writes_artifacts(sample_df, tmp_path):
    results = run_training(
        sample_df.head(900), model_path=tmp_path / "model.joblib", metrics_path=tmp_path / "metrics.json",
        n_search_iter=2, shap_image_path=tmp_path / "shap.png", report_text_path=tmp_path / "report.txt")
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "shap.png").stat().st_size > 1000
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["deployed_model"] == "XGBoost"
    assert set(saved["model_comparison"]) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert results["dataset"]["train_rows"] + results["dataset"]["test_rows"] == 900
    reloaded = joblib.load(tmp_path / "model.joblib")
    assert hasattr(reloaded, "predict_proba")


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_saved_model_and_metrics_are_consistent():
    metrics = json.loads(METRICS_PATH.read_text())
    deployed = metrics["model_comparison"][metrics["deployed_model"]]
    assert 0.5 < deployed["test_roc_auc"] < 1.0
    assert metrics["dataset"]["train_rows"] + metrics["dataset"]["test_rows"] == metrics["dataset"]["rows"]
