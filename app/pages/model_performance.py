"""Page 5: Model Performance. Shows only results saved by the training pipeline."""
import pandas as pd
import streamlit as st

from components import charts, ui
from components.data_access import data_source_pills, load_metrics
from components.theme import BLUE
from src.config import SHAP_SUMMARY_IMAGE_PATH

metrics = load_metrics()
ui.page_header(
    "Model Performance",
    "How the models were compared and how the deployed XGBoost model performs on customers it never saw during training.",
    data_source_pills(),
)
if metrics is None:
    st.warning("No evaluation results found. Train the model first with `python -m src.train`.")
    st.stop()

deployed = metrics["deployed_model"]
scores = metrics["model_comparison"][deployed]
ui.kpi_row([
    dict(label="Accuracy", value=f"{scores['test_accuracy']:.3f}"),
    dict(label="Precision", value=f"{scores['test_precision']:.3f}", caption="Of predicted churners, share who churn"),
    dict(label="Recall", value=f"{scores['test_recall']:.3f}", caption="Of actual churners, share found"),
    dict(label="F1-score", value=f"{scores['test_f1']:.3f}"),
    dict(label="ROC-AUC", value=f"{scores['test_roc_auc']:.3f}", tone="accent"),
])
st.caption(f"{deployed}, evaluated on {metrics['dataset']['test_rows']:,} held-out customers at a "
           f"{metrics['threshold']:.0%} probability threshold.")

tab_compare, tab_eval, tab_explain, tab_details = st.tabs(
    ["Model comparison", "Evaluation", "Explainability", "Training details"])

with tab_compare:
    comparison = metrics["model_comparison"]
    table = pd.DataFrame(comparison).T.rename(columns={
        "cv_roc_auc_mean": "CV ROC-AUC (mean)", "cv_roc_auc_std": "CV ROC-AUC (std)", "test_accuracy": "Accuracy",
        "test_precision": "Precision", "test_recall": "Recall", "test_f1": "F1-score", "test_roc_auc": "Test ROC-AUC"})
    table.index.name = "Model"
    st.dataframe(table.round(4), width="stretch")
    left, right = st.columns([1.4, 1])
    with left:
        ui.chart_card(charts.metric_comparison(comparison), key="perf_compare")
    with right:
        with ui.card():
            ui.section_title("How the model was chosen")
            best = metrics["best_by_criterion"]
            st.markdown(f"Models are ranked by **{metrics['selection_criterion']}**, so the test set plays no part in the ranking.")
            st.markdown(f"Top-ranked: **{best}**. Deployed: **{deployed}**.")
            ui.callout("The models score very close to each other. Differences are about the size of the "
                       "cross-validation spread, so none is clearly better. XGBoost was also tuned on the "
                       "training data while the others use fixed settings.", "amber")

with tab_eval:
    left, right = st.columns(2)
    with left:
        ui.chart_card(charts.confusion_matrix_figure(metrics["evaluation"][deployed]["confusion_matrix"]), key="perf_cm")
    with right:
        ui.chart_card(charts.roc_figure(metrics["evaluation"], comparison, deployed), key="perf_roc")
    report = pd.DataFrame(metrics["evaluation"][deployed]["classification_report"]).T
    with st.expander("Full classification report"):
        st.dataframe(report.round(3), width="stretch")
    ui.callout("Class weighting makes the model favour <b>recall</b>: it finds more churners but also flags more "
               "customers who would have stayed. The 50% threshold can be adjusted to match the cost of a missed "
               "churner versus an unnecessary retention offer.")

with tab_explain:
    left, right = st.columns(2)
    with left:
        ui.chart_card(charts.importance_bar(metrics["shap_importance"], "mean_abs_shap",
                                            "Mean absolute SHAP value (test set)"), key="perf_shap")
    with right:
        ui.chart_card(charts.importance_bar(metrics["xgboost_gain_importance"], "importance",
                                            "XGBoost gain importance", color=BLUE), key="perf_gain")
    with ui.card():
        ui.section_title("SHAP summary", "Each dot is a customer. Red means a high feature value, blue a low one.")
        if SHAP_SUMMARY_IMAGE_PATH.exists():
            st.image(str(SHAP_SUMMARY_IMAGE_PATH), width="stretch")
        else:
            st.info("The SHAP summary image is missing. Re-run `python -m src.train`.")
    ui.callout("<b>Reading importance with care.</b> Importance shows what the model relies on, not what causes churn. "
               "Correlated features can share or hide credit (for example tenure and total charges), and one-hot "
               "columns of one feature are summed here. Use explanations to generate hypotheses to check, not conclusions.")

with tab_details:
    dataset = metrics["dataset"]
    left, right = st.columns(2)
    with left:
        with ui.card():
            ui.section_title("Training data")
            st.dataframe(pd.DataFrame({
                "Item": ["Customers", "Model features", "Training rows", "Test rows", "Churn rate (train / test)",
                         "Class weight (scale_pos_weight)", "Random seed"],
                "Value": [f"{dataset['rows']:,}", dataset["n_features"], f"{dataset['train_rows']:,}", f"{dataset['test_rows']:,}",
                          f"{dataset['train_churn_rate']:.1%} / {dataset['test_churn_rate']:.1%}",
                          f"{dataset['scale_pos_weight']:.2f}", dataset["random_state"]],
            }).astype(str), hide_index=True, width="stretch")
        with ui.card():
            ui.section_title("Library versions")
            st.dataframe(pd.DataFrame(metrics["versions"].items(), columns=["Library", "Version"]),
                         hide_index=True, width="stretch")
    with right:
        with ui.card():
            ui.section_title("Tuned XGBoost parameters", "Chosen by randomized search on the training set only.")
            st.dataframe(pd.DataFrame(metrics["xgboost_best_params"].items(), columns=["Parameter", "Value"]).astype(str),
                         hide_index=True, width="stretch")
        with ui.card():
            ui.section_title("Model limitations")
            ui.bullet_list([
                "Trained on one sample dataset; it may not transfer to another business or period.",
                "Churn is observed once per customer, so the model cannot say when a customer will leave.",
                "Probabilities are not calibrated because of class weighting.",
                "Explanations show associations, not causes.",
                "No fairness audit has been done beyond leaving out the gender column.",
            ])
    if metrics["dataset"]["is_synthetic"]:
        ui.callout("This model was trained on a <b>synthetic</b> dataset, so these results say nothing about real customers.", "amber")
