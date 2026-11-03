"""Page 1: Executive Dashboard."""
import streamlit as st

from components import charts, ui
from components.data_access import data_source_pills, load_customer_data
from src.eda import SEGMENT_ORDER, churn_rate_by, generate_insights, overall_summary

df = load_customer_data()
summary = overall_summary(df)

ui.page_header(
    "Executive Dashboard",
    "A snapshot of churn across the analysed customer base: how many customers left, who they are and where to look first.",
    data_source_pills() + [(f"{summary['total_customers']:,} customers", "")],
)

ui.kpi_row([
    dict(label="Total customers", value=f"{summary['total_customers']:,}", caption="In the loaded dataset"),
    dict(label="Churned customers", value=f"{summary['churned']:,}", caption="Left the service", tone="danger"),
    dict(label="Retained customers", value=f"{summary['retained']:,}", caption="Still subscribed", tone="accent"),
    dict(label="Overall churn rate", value=f"{summary['churn_rate']:.1%}", caption="Churned / total customers"),
    dict(label="Avg monthly charges", value=f"{summary['avg_monthly_charges']:,.2f}",
         caption=f"Average tenure {summary['avg_tenure']:.0f} months"),
])

st.write("")
left, middle, right = st.columns([1, 1.25, 1.25])
with left:
    ui.chart_card(charts.churn_donut(df), key="exec_donut")
with middle:
    ui.chart_card(charts.churn_rate_bar(df, "Contract", "Churn rate by contract type",
                                        order=["Month-to-month", "One year", "Two year"]), key="exec_contract")
with right:
    ui.chart_card(charts.churn_rate_bar(df, "TenureBand", "Churn rate by tenure", order=list(df["TenureBand"].cat.categories)),
                  key="exec_tenure")

ui.section_title("Spending and segments", "How billing and customer type relate to churn.")
left, right = st.columns(2)
with left:
    ui.chart_card(charts.distribution_by_status(df, "MonthlyCharges", "Monthly charges: churned vs retained",
                                                "Monthly charges", bins=35), key="exec_monthly")
with right:
    ui.chart_card(charts.churn_rate_bar(df, "Segment", "Churn rate by customer segment", order=SEGMENT_ORDER),
                  key="exec_segments")

segment_table = churn_rate_by(df, "Segment")
with st.expander("Segment definitions and sizes"):
    st.caption("Segments are simple rules on contract and tenure, so they are easy to explain and reproduce.")
    st.dataframe(
        segment_table.rename(columns={
            "customers": "Customers", "churned": "Churned", "churn_rate": "Churn rate",
            "avg_tenure": "Avg tenure (months)", "avg_monthly_charges": "Avg monthly charges"}),
        hide_index=True, width="stretch",
        column_config={"Churn rate": st.column_config.NumberColumn(format="percent"),
                       "Avg tenure (months)": st.column_config.NumberColumn(format="%.1f"),
                       "Avg monthly charges": st.column_config.NumberColumn(format="%.2f")},
    )

ui.section_title("Key observations", "Computed from the loaded data. These are associations, not proof of cause.")
with ui.card():
    ui.bullet_list(generate_insights(df))
