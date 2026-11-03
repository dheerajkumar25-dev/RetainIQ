"""Model evaluation helpers. Every number shown in the app comes from these functions."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CLASSIFICATION_THRESHOLD
from src.preprocessing import map_encoded_to_original


def compute_classification_metrics(y_true, y_proba, threshold: float = CLASSIFICATION_THRESHOLD) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def compute_confusion_matrix(y_true, y_proba, threshold: float = CLASSIFICATION_THRESHOLD) -> dict:
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def compute_roc_curve(y_true, y_proba) -> dict:
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_proba)
    return {"fpr": np.round(false_positive_rate, 4).tolist(), "tpr": np.round(true_positive_rate, 4).tolist()}


def compute_classification_report(y_true, y_proba, threshold: float = CLASSIFICATION_THRESHOLD):
    y_pred = (np.asarray(y_proba) >= threshold).astype(int)
    kwargs = dict(target_names=["Retained", "Churned"], zero_division=0)
    return (classification_report(y_true, y_pred, output_dict=True, **kwargs),
            classification_report(y_true, y_pred, **kwargs))


def evaluate_pipeline(pipeline, features: pd.DataFrame, target, threshold: float = CLASSIFICATION_THRESHOLD) -> dict:
    """Score a fitted pipeline on held-out data and return all evaluation artefacts."""
    probabilities = pipeline.predict_proba(features)[:, 1]
    report_dict, report_text = compute_classification_report(target, probabilities, threshold)
    return {
        "metrics": compute_classification_metrics(target, probabilities, threshold),
        "confusion_matrix": compute_confusion_matrix(target, probabilities, threshold),
        "roc_curve": compute_roc_curve(target, probabilities),
        "classification_report": report_dict,
        "classification_report_text": report_text,
    }


def get_xgboost_feature_importance(pipeline) -> pd.DataFrame:
    """Gain-based importance from XGBoost, summed over one-hot columns of the same original feature."""
    encoded_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_
    table = pd.DataFrame({
        "feature": map_encoded_to_original(encoded_names),
        "importance": importances,
    })
    table = table.groupby("feature", as_index=False)["importance"].sum()
    return table.sort_values("importance", ascending=False).reset_index(drop=True)
