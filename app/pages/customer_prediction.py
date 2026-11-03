"""Page 2: Customer Prediction (single customer)."""
import re

import streamlit as st

from components import charts, ui
from components.data_access import data_source_pills, get_explainer, get_pipeline
from src.config import CATEGORY_OPTIONS, INTERNET_ADD_ONS
from src.explainability import explain_single_customer
from src.predict import predict_single
from src.recommendations import suggest_retention_actions
from src.validation import InvalidInputError

pipeline = get_pipeline()
explainer = get_explainer()

ui.page_header(
    "Customer Prediction",
    "Describe a customer to estimate their probability of churning and see which factors drive the estimate.",
    data_source_pills(),
)

form_column, result_column = st.columns([1.15, 1], gap="large")

# ------------------------------------------------------------------ inputs
with form_column:
    customer = {}
    with ui.card():
        ui.section_title("Account and billing")
        customer["tenure"] = st.slider("Tenure (months)", 0, 72, 12, key="in_tenure",
                                       help="How long the customer has been subscribed.")
        col_a, col_b = st.columns(2)
        customer["MonthlyCharges"] = col_a.number_input("Monthly charges", 0.0, 500.0, 70.0, 1.0, format="%.2f", key="in_monthly")
        customer["TotalCharges"] = col_b.number_input("Total charges", 0.0, 100000.0, 840.0, 10.0, format="%.2f", key="in_total",
                                                      help="Lifetime billing. Roughly tenure x monthly charges.")
        col_c, col_d = st.columns(2)
        customer["Contract"] = col_c.selectbox("Contract", CATEGORY_OPTIONS["Contract"], key="in_contract")
        customer["PaymentMethod"] = col_d.selectbox("Payment method", CATEGORY_OPTIONS["PaymentMethod"], key="in_payment")
        customer["PaperlessBilling"] = st.selectbox("Paperless billing", CATEGORY_OPTIONS["PaperlessBilling"], index=1, key="in_paperless")

    with ui.card():
        ui.section_title("Services")
        col_e, col_f = st.columns(2)
        customer["InternetService"] = col_e.selectbox("Internet service", CATEGORY_OPTIONS["InternetService"], index=1, key="in_internet")
        customer["PhoneService"] = col_f.selectbox("Phone service", CATEGORY_OPTIONS["PhoneService"], index=1, key="in_phone")

        if customer["PhoneService"] == "No":
            st.text_input("Multiple lines", "No phone service", disabled=True, key="in_lines_na")
            customer["MultipleLines"] = "No phone service"
        else:
            customer["MultipleLines"] = st.selectbox("Multiple lines", ["No", "Yes"], key="in_lines")

        add_on_columns = st.columns(3)
        for position, name in enumerate(INTERNET_ADD_ONS):
            with add_on_columns[position % 3]:
                label = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)  # "TechSupport" -> "Tech Support"
                if customer["InternetService"] == "No":
                    st.text_input(label, "No internet service", disabled=True, key=f"in_{name}_na")
                    customer[name] = "No internet service"
                else:
                    customer[name] = st.selectbox(label, ["No", "Yes"], key=f"in_{name}")

    with ui.card():
        ui.section_title("Customer profile")
        col_g, col_h, col_i = st.columns(3)
        customer["SeniorCitizen"] = col_g.selectbox("Senior citizen", CATEGORY_OPTIONS["SeniorCitizen"], key="in_senior")
        customer["Partner"] = col_h.selectbox("Partner", CATEGORY_OPTIONS["Partner"], key="in_partner")
        customer["Dependents"] = col_i.selectbox("Dependents", CATEGORY_OPTIONS["Dependents"], key="in_dependents")

    predict_clicked = st.button("Predict churn risk", type="primary", width="stretch")

if predict_clicked:
    try:
        prediction = predict_single(pipeline, customer)
        drivers = explain_single_customer(pipeline, prediction["features"], explainer, top_n=8)
        st.session_state["prediction"] = {
            "inputs": dict(customer),
            "prediction": prediction,
            "drivers": drivers,
            "actions": suggest_retention_actions(customer, drivers),
        }
    except InvalidInputError as error:
        st.session_state.pop("prediction", None)
        with form_column:
            for message in error.errors:
                st.error(message)

# ------------------------------------------------------------------ result
with result_column:
    saved = st.session_state.get("prediction")
    if saved is None:
        st.markdown(
            '<div class="riq-placeholder"><b>No prediction yet</b><br>Fill in the customer details and choose '
            '"Predict churn risk". The estimate and its explanation appear here.</div>', unsafe_allow_html=True)
    else:
        prediction = saved["prediction"]
        probability, risk = prediction["churn_probability"], prediction["risk_category"]

        if saved["inputs"] != customer:
            ui.callout("The inputs changed since this prediction. Choose <b>Predict churn risk</b> to update it.", "amber")

        with ui.card():
            st.plotly_chart(charts.probability_gauge(probability, risk), width="stretch", config=ui.PLOT_CONFIG, key="gauge")
            tone = {"Low": "teal", "Medium": "amber", "High": "coral"}[risk]
            st.markdown(
                f'<div class="riq-verdict risk-{risk.lower()}"><div class="riq-verdict-label">{prediction["predicted_class"]}</div>'
                f'<div class="riq-verdict-sub">Estimated churn probability {probability:.1%}</div>'
                f'<div style="margin-top:0.7rem">{ui.pill(f"{risk} risk", tone)}'
                f'{ui.pill("Threshold 50%")}</div></div>', unsafe_allow_html=True)

        for warning in prediction["warnings"]:
            st.warning(warning)

        ui.chart_card(charts.driver_bar(saved["drivers"]), key="drivers")

        with ui.card():
            ui.section_title("Retention considerations")
            if risk != "Low" and saved["actions"]:
                ui.bullet_list(saved["actions"])
            elif risk == "Low":
                st.write("The model sees low risk for this customer. Regular engagement is probably enough.")
            else:
                st.write("No specific rule applies to the main risk drivers. Review the chart above for context.")

        ui.callout(
            "<b>This is an estimate, not a certainty.</b> The probability comes from patterns in the training data. "
            "Because the model uses class weighting, treat it as a risk score for ranking customers rather than an exact chance. "
            "Use it to support, not replace, human judgement.")
