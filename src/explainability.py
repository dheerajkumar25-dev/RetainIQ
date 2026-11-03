"""SHAP-based explanations for the XGBoost pipeline.

SHAP values are expressed in log-odds (the model's raw output). A positive value
pushes a customer towards "churn", a negative value towards "stay". One-hot
columns of the same original feature are summed so the result is readable
("Contract" instead of three separate columns); this is valid because SHAP is additive.
"""
import matplotlib

matplotlib.use("Agg")  # no GUI needed; works on servers and Windows alike
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import SHAP_SUMMARY_IMAGE_PATH
from src.preprocessing import map_encoded_to_original

DARK_BACKGROUND = "#111C33"
LIGHT_TEXT = "#DCE6F5"


def build_explainer(pipeline) -> "shap.TreeExplainer":
    return shap.TreeExplainer(pipeline.named_steps["model"])


def transform_features(pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted preprocessor and keep readable column names."""
    preprocessor = pipeline.named_steps["preprocessor"]
    encoded = preprocessor.transform(features)
    names = [name.split("__", 1)[1] for name in preprocessor.get_feature_names_out()]
    return pd.DataFrame(encoded, columns=names, index=features.index)


def compute_shap_values(pipeline, features: pd.DataFrame, explainer=None):
    """Return (shap values per encoded column, base value, encoded features)."""
    explainer = explainer or build_explainer(pipeline)
    encoded = transform_features(pipeline, features)
    values = np.asarray(explainer.shap_values(encoded))
    base_value = float(np.ravel(explainer.expected_value)[0])
    return values, base_value, encoded


def aggregate_by_original_feature(shap_values: np.ndarray, encoded_columns) -> pd.DataFrame:
    """Sum SHAP values of one-hot columns that belong to the same original feature."""
    original = map_encoded_to_original(list(encoded_columns))
    frame = pd.DataFrame(shap_values, columns=encoded_columns)
    return frame.T.groupby(original).sum().T


def explain_single_customer(pipeline, customer: pd.DataFrame, explainer=None, top_n: int = 8) -> pd.DataFrame:
    """Top feature contributions for one customer (a one-row DataFrame)."""
    values, _, encoded = compute_shap_values(pipeline, customer, explainer)
    by_feature = aggregate_by_original_feature(values, encoded.columns).iloc[0]
    table = pd.DataFrame({
        "feature": by_feature.index,
        "value": [customer.iloc[0][name] for name in by_feature.index],
        "shap_value": by_feature.values,
    })
    order = table["shap_value"].abs().sort_values(ascending=False).index
    return table.loc[order].head(top_n).reset_index(drop=True)


def global_importance(pipeline, features: pd.DataFrame, explainer=None) -> pd.DataFrame:
    """Mean absolute SHAP value per original feature."""
    values, _, encoded = compute_shap_values(pipeline, features, explainer)
    by_feature = aggregate_by_original_feature(values, encoded.columns)
    table = by_feature.abs().mean().rename("mean_abs_shap").reset_index()
    table.columns = ["feature", "mean_abs_shap"]
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def save_shap_summary_plot(pipeline, features: pd.DataFrame, path=SHAP_SUMMARY_IMAGE_PATH,
                           max_display: int = 15, explainer=None) -> None:
    """Save the SHAP beeswarm plot (per encoded column) styled to match the dashboard."""
    values, _, encoded = compute_shap_values(pipeline, features, explainer)
    plt.figure(figsize=(9, 6.2))
    shap.summary_plot(values, encoded, max_display=max_display, show=False, plot_size=None)
    figure = plt.gcf()
    figure.patch.set_facecolor(DARK_BACKGROUND)
    for axis in figure.axes:
        axis.set_facecolor(DARK_BACKGROUND)
        axis.tick_params(colors=LIGHT_TEXT)
        axis.xaxis.label.set_color(LIGHT_TEXT)
        axis.yaxis.label.set_color(LIGHT_TEXT)
        for spine in axis.spines.values():
            spine.set_color("#2B3A57")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, bbox_inches="tight", facecolor=DARK_BACKGROUND)
    plt.close(figure)
