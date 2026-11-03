"""Train, compare and save the churn models.

Run with:  python -m src.train

Workflow
1. Load + clean the dataset.
2. Stratified train/test split (the test set is not touched until the end).
3. Build three pipelines (preprocessor + model) so imputing/scaling/encoding is
   fitted on training data only -- also inside every cross-validation fold.
4. Tune XGBoost with a small randomized search (training data only).
5. Compare all models by 5-fold cross-validated ROC-AUC on the training set.
6. Evaluate on the untouched test set and save the XGBoost pipeline, metrics and SHAP plots.
"""
import argparse
import json
import platform
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
import xgboost
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    CLASSIFICATION_REPORT_PATH,
    CLASSIFICATION_THRESHOLD,
    CV_FOLDS,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    SHAP_SUMMARY_IMAGE_PATH,
    TEST_SIZE,
    TUNING_ITERATIONS,
)
from src.data_loader import is_synthetic_dataset, load_clean_dataset
from src.evaluate import evaluate_pipeline, get_xgboost_feature_importance
from src.explainability import build_explainer, global_importance, save_shap_summary_plot
from src.preprocessing import build_preprocessor, make_train_test_split

PRIMARY_MODEL = "XGBoost"


def build_pipelines(scale_pos_weight: float) -> dict:
    """One pipeline per candidate model. Imbalance is handled by class weights."""
    def make(model):
        return Pipeline([("preprocessor", build_preprocessor()), ("model", model)])

    return {
        "Logistic Regression": make(LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        "Random Forest": make(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE)),
        PRIMARY_MODEL: make(XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE)),
    }


def tune_xgboost(pipeline: Pipeline, features, target, n_iter: int):
    """Small randomized search on the training set only. Returns (best pipeline, best params)."""
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "model__n_estimators": randint(100, 500),
            "model__max_depth": randint(2, 6),
            "model__learning_rate": uniform(0.02, 0.18),
            "model__subsample": uniform(0.7, 0.3),
            "model__colsample_bytree": uniform(0.6, 0.4),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": uniform(1, 9),
        },
        n_iter=n_iter, scoring="roc_auc",
        cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    search.fit(features, target)
    best_params = {key.replace("model__", ""): (float(v) if isinstance(v, (float, np.floating)) else int(v))
                   for key, v in search.best_params_.items()}
    return search.best_estimator_, best_params


def cross_validated_auc(pipeline, features, target) -> tuple[float, float]:
    folds = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipeline, features, target, cv=folds, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def run_training(df: pd.DataFrame, model_path=MODEL_PATH, metrics_path=METRICS_PATH,
                 n_search_iter: int = TUNING_ITERATIONS, shap_image_path=SHAP_SUMMARY_IMAGE_PATH,
                 report_text_path=CLASSIFICATION_REPORT_PATH, data_is_synthetic: bool = False) -> dict:
    X_train, X_test, y_train, y_test = make_train_test_split(df)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    pipelines = build_pipelines(scale_pos_weight)

    print(f"Train rows: {len(X_train):,}  Test rows: {len(X_test):,}  scale_pos_weight: {scale_pos_weight:.2f}")
    print(f"Tuning XGBoost ({n_search_iter} random configurations)...")
    pipelines[PRIMARY_MODEL], xgb_params = tune_xgboost(pipelines[PRIMARY_MODEL], X_train, y_train, n_search_iter)

    comparison, test_results = {}, {}
    for name, pipeline in pipelines.items():
        print(f"Evaluating {name}...")
        cv_mean, cv_std = cross_validated_auc(pipeline, X_train, y_train)
        pipeline.fit(X_train, y_train)  # CV clones the pipeline, so fit the final one on all training rows
        test_results[name] = evaluate_pipeline(pipeline, X_test, y_test)
        comparison[name] = {"cv_roc_auc_mean": cv_mean, "cv_roc_auc_std": cv_std,
                            **{f"test_{k}": v for k, v in test_results[name]["metrics"].items()}}

    ranking = sorted(comparison, key=lambda m: comparison[m]["cv_roc_auc_mean"], reverse=True)
    best_by_criterion = ranking[0]
    gap = comparison[best_by_criterion]["cv_roc_auc_mean"] - comparison[PRIMARY_MODEL]["cv_roc_auc_mean"]
    print(f"Ranking by CV ROC-AUC: {ranking}. Deployed model: {PRIMARY_MODEL} (gap to best: {gap:.4f})")

    deployed = pipelines[PRIMARY_MODEL]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed, model_path)

    explainer = build_explainer(deployed)
    shap_table = global_importance(deployed, X_test, explainer)
    save_shap_summary_plot(deployed, X_test, shap_image_path, explainer=explainer)
    gain_table = get_xgboost_feature_importance(deployed)

    results = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "deployed_model": PRIMARY_MODEL,
        "selection_criterion": f"{CV_FOLDS}-fold cross-validated ROC-AUC on the training set",
        "ranking_by_criterion": ranking,
        "best_by_criterion": best_by_criterion,
        "deployed_gap_to_best_cv_auc": float(gap),
        "threshold": CLASSIFICATION_THRESHOLD,
        "dataset": {
            "rows": int(len(df)), "n_features": len(FEATURE_COLUMNS), "is_synthetic": data_is_synthetic,
            "train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": TEST_SIZE,
            "train_churn_rate": float(y_train.mean()), "test_churn_rate": float(y_test.mean()),
            "scale_pos_weight": scale_pos_weight, "random_state": RANDOM_STATE,
        },
        "xgboost_best_params": xgb_params,
        "model_comparison": comparison,
        "evaluation": {name: {k: v for k, v in res.items() if k != "classification_report_text"}
                       for name, res in test_results.items()},
        "shap_importance": shap_table.round(5).to_dict(orient="records"),
        "xgboost_gain_importance": gain_table.round(5).to_dict(orient="records"),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "xgboost": xgboost.__version__, "shap": shap.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_text_path.write_text(
        f"Classification report ({PRIMARY_MODEL}, test set, threshold {CLASSIFICATION_THRESHOLD})\n\n"
        + test_results[PRIMARY_MODEL]["classification_report_text"], encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RetainIQ churn model.")
    parser.add_argument("--search-iterations", type=int, default=TUNING_ITERATIONS,
                        help="Number of random XGBoost configurations to try.")
    args = parser.parse_args()

    df = load_clean_dataset()
    results = run_training(df, n_search_iter=args.search_iterations, data_is_synthetic=is_synthetic_dataset())

    print("\nTest-set results (held-out 20%):")
    table = pd.DataFrame(results["model_comparison"]).T
    print(table.round(4).to_string())
    print(f"\nSaved model:   {MODEL_PATH}\nSaved metrics: {METRICS_PATH}")


if __name__ == "__main__":
    main()
