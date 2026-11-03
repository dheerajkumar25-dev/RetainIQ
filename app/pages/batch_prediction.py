"""Page 3: Batch Prediction from a CSV upload."""
import pandas as pd
import streamlit as st

from components import charts, ui
from components.data_access import data_source_pills, get_pipeline, get_sample_template
from src.config import FEATURE_COLUMNS, MAX_BATCH_ROWS
from src.predict import LIKELY_TO_CHURN, predict_batch
from src.validation import validate_customer_frame

pipeline = get_pipeline()

ui.page_header(
    "Batch Prediction",
    "Upload a CSV of customers to score them all at once, then download the results.",
    data_source_pills(),
)

with ui.card():
    info_column, download_column = st.columns([2.2, 1])
    with info_column:
        ui.section_title("Required columns")
        st.markdown(", ".join(f"`{name}`" for name in FEATURE_COLUMNS))
        st.caption(f"Extra columns such as customerID are kept in the output. Maximum {MAX_BATCH_ROWS:,} rows. "
                   "Empty cells are allowed and filled using training-set statistics.")
    with download_column:
        st.write("")
        st.download_button("Download sample CSV", get_sample_template(), "retainiq_sample_customers.csv",
                           "text/csv", width="stretch", icon=":material/download:")

uploaded = st.file_uploader("Upload customer CSV", type=["csv"], help="Comma-separated, one customer per row.")
if uploaded is None:
    st.info("Upload a CSV file to begin, or download the sample above to see the expected format.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
except pd.errors.EmptyDataError:
    st.error("The file is empty.")
    st.stop()
except (pd.errors.ParserError, UnicodeDecodeError) as error:
    st.error(f"The file could not be read as a CSV: {error}")
    st.stop()

ui.section_title("Data preview", f"{len(raw):,} rows and {raw.shape[1]} columns.")
st.dataframe(raw.head(10), width="stretch", hide_index=True)

validation = validate_customer_frame(raw)
if not validation.is_valid:
    for message in validation.errors:
        st.error(message)
    st.stop()
if validation.warnings:
    with st.expander(f"{len(validation.warnings)} warning(s) about this file"):
        for message in validation.warnings:
            st.warning(message)
else:
    st.success("Validation passed: all required columns and values look correct.")

file_signature = (uploaded.name, uploaded.size)
if st.button("Run batch prediction", type="primary"):
    with st.spinner("Scoring customers..."):
        results = predict_batch(pipeline, raw)
    st.session_state["batch_results"] = {"signature": file_signature, "results": results}

saved = st.session_state.get("batch_results")
if not saved or saved["signature"] != file_signature:
    st.stop()

results = saved["results"]
churn_count = int((results["predicted_class"] == LIKELY_TO_CHURN).sum())
high_count = int((results["risk_category"] == "High").sum())

ui.section_title("Results")
ui.kpi_row([
    dict(label="Customers scored", value=f"{len(results):,}"),
    dict(label="Likely to churn", value=f"{churn_count:,}", caption=f"{churn_count / len(results):.1%} of customers", tone="danger"),
    dict(label="High-risk customers", value=f"{high_count:,}", caption="Probability of 65% or more"),
    dict(label="Average probability", value=f"{results['churn_probability'].mean():.1%}", caption="Across all customers"),
])
st.write("")
left, right = st.columns(2)
with left:
    ui.chart_card(charts.probability_histogram(results["churn_probability"]), key="batch_hist")
with right:
    ui.chart_card(charts.risk_bar(results["risk_category"]), key="batch_risk")

ordered = results.sort_values("churn_probability", ascending=False)
risk_filter = st.multiselect("Show risk categories", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
shown = ordered[ordered["risk_category"].isin(risk_filter)]
leading = [c for c in ["customerID", "churn_probability", "risk_category", "predicted_class"] if c in shown.columns]
shown = shown[leading + [c for c in shown.columns if c not in leading]]
st.dataframe(
    shown, width="stretch", hide_index=True, height=420,
    column_config={"churn_probability": st.column_config.ProgressColumn(
        "Churn probability", min_value=0.0, max_value=1.0, format="percent")},
)
st.download_button("Download predictions (CSV)", ordered.to_csv(index=False).encode("utf-8"),
                   "retainiq_predictions.csv", "text/csv", type="primary", icon=":material/download:")
ui.callout("Probabilities are model estimates and are not calibrated; use them to rank customers, not as exact chances.")
