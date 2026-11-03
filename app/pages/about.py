"""Page 6: About the project."""
import pandas as pd
import streamlit as st

from components import ui
from components.data_access import data_source_pills, load_metrics
from src.config import DATA_DICTIONARY, EXCLUDED_COLUMNS

metrics = load_metrics()
ui.page_header(
    "About RetainIQ",
    "An end-to-end churn prediction project: data preparation, model comparison, explainability and a dashboard to use it.",
    data_source_pills(),
)

left, right = st.columns(2)
with left:
    with ui.card():
        ui.section_title("The business problem")
        st.write("Winning a new subscriber usually costs more than keeping an existing one. If a team can spot "
                 "customers who are likely to leave, it can focus its retention effort where it matters. "
                 "RetainIQ estimates churn risk for each customer and shows the factors behind the estimate.")
with right:
    with ui.card():
        ui.section_title("The dataset")
        rows = f"{metrics['dataset']['rows']:,}" if metrics else "7,043"
        st.write(f"The IBM Telco Customer Churn sample dataset ({rows} customers). IBM publishes it to demonstrate "
                 "analytics tools, so treat it as a benchmark for the method rather than verified records from a "
                 "specific company. If it cannot be downloaded, the project can generate a synthetic stand-in that "
                 "is always labelled as such.")

ui.section_title("Machine learning workflow")
steps = [
    ("Load and clean", "Check types, missing values and duplicates. Blank TotalCharges for brand-new customers become 0."),
    ("Split", "Stratified 80/20 train-test split so both parts share the same churn rate."),
    ("Preprocess", "Impute, scale numeric columns and one-hot encode categories, fitted on training data only."),
    ("Train and compare", "Logistic Regression, Random Forest and a tuned XGBoost model with class weighting."),
    ("Evaluate", "Accuracy, precision, recall, F1, ROC-AUC and a confusion matrix on the held-out test set."),
    ("Explain and save", "SHAP explanations, then the full pipeline is saved with joblib and reused by this app."),
]
rows_of_steps = [steps[:3], steps[3:]]
counter = 1
for row in rows_of_steps:
    for column, (title, text) in zip(st.columns(3), row):
        with column, ui.card():
            st.markdown(f"**{counter}. {title}**")
            st.caption(text)
        counter += 1

left, right = st.columns(2)
with left:
    with ui.card():
        ui.section_title("Technology")
        st.write("Python, Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Joblib, Matplotlib, Seaborn, Plotly, Streamlit and Pytest.")
    with ui.card():
        ui.section_title("Known limitations")
        ui.bullet_list([
            "One dataset, one time period: no check on how the model ages.",
            "Probabilities are risk scores, not calibrated chances.",
            "Explanations describe the model, not real-world causes.",
            "Retention suggestions are simple rules, not tested interventions.",
        ])
with right:
    with ui.card():
        ui.section_title("Possible next steps")
        ui.bullet_list([
            "Probability calibration and a business-cost-based decision threshold.",
            "Time-based validation and monitoring for data drift.",
            "Fairness analysis across customer groups.",
            "Uplift modelling to estimate which offers actually retain customers.",
            "A REST API for the model alongside the dashboard.",
        ])

with st.expander("Data dictionary"):
    dictionary = pd.DataFrame(
        [(name, kind, role, description) for name, (kind, role, description) in DATA_DICTIONARY.items()],
        columns=["Column", "Type", "Role", "Description"])
    st.dataframe(dictionary, hide_index=True, width="stretch")
    st.caption("Left out of the model: " + " ".join(f"{name}: {reason}" for name, reason in EXCLUDED_COLUMNS.items()))
