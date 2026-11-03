"""Page 4: Customer Analytics with interactive filters."""
import streamlit as st

from components import charts, ui
from components.data_access import data_source_pills, load_customer_data
from src.eda import SEGMENT_ORDER, churn_rate_by

df = load_customer_data()

ui.page_header(
    "Customer Analytics",
    "Explore how churn varies across contracts, services, billing and customer segments. Filters apply to every chart.",
    data_source_pills(),
)

with ui.card():
    col_a, col_b, col_c, col_d = st.columns([1.2, 1.2, 1.2, 0.9])
    contracts = col_a.multiselect("Contract", sorted(df["Contract"].unique()), default=sorted(df["Contract"].unique()))
    internet = col_b.multiselect("Internet service", sorted(df["InternetService"].unique()), default=sorted(df["InternetService"].unique()))
    tenure_min, tenure_max = int(df["tenure"].min()), int(df["tenure"].max())
    tenure_range = col_c.slider("Tenure (months)", tenure_min, tenure_max, (tenure_min, tenure_max))
    senior = col_d.selectbox("Senior citizen", ["All", "Yes", "No"])

mask = df["Contract"].isin(contracts) & df["InternetService"].isin(internet) & df["tenure"].between(*tenure_range)
if senior != "All":
    mask &= df["SeniorCitizen"] == senior
filtered = df[mask]

if filtered.empty:
    st.warning("No customers match these filters. Widen the filters to see results.")
    st.stop()

ui.kpi_row([
    dict(label="Customers in view", value=f"{len(filtered):,}", caption=f"{len(filtered) / len(df):.0%} of all customers"),
    dict(label="Churn rate", value=f"{filtered['ChurnFlag'].mean():.1%}", caption=f"All customers: {df['ChurnFlag'].mean():.1%}", tone="danger"),
    dict(label="Average tenure", value=f"{filtered['tenure'].mean():.1f} mo"),
    dict(label="Avg monthly charges", value=f"{filtered['MonthlyCharges'].mean():,.2f}"),
])
if len(filtered) < 100:
    ui.callout("Fewer than 100 customers are in view, so rates can swing a lot. Interpret with care.", "amber")
st.write("")

drivers_tab, distributions_tab, segments_tab = st.tabs(["Churn drivers", "Distributions", "Segments"])

with drivers_tab:
    left, right = st.columns(2)
    with left:
        ui.chart_card(charts.churn_rate_bar(filtered, "Contract", "Churn by contract type"), key="an_contract")
    with right:
        ui.chart_card(charts.churn_rate_bar(filtered, "InternetService", "Churn by internet service"), key="an_internet")
    ui.chart_card(charts.churn_rate_bar(filtered, "PaymentMethod", "Churn by payment method", height=300), key="an_payment")

with distributions_tab:
    left, right = st.columns(2)
    with left:
        ui.chart_card(charts.distribution_by_status(filtered, "tenure", "Tenure distribution", "Tenure (months)", bins=24), key="an_tenure")
    with right:
        ui.chart_card(charts.distribution_by_status(filtered, "MonthlyCharges", "Monthly charges distribution", "Monthly charges", bins=30), key="an_monthly")
    st.caption("Each colour is scaled to 100% so the shapes can be compared even though far fewer customers churned.")

with segments_tab:
    left, right = st.columns(2)
    with left:
        ui.chart_card(charts.churn_rate_bar(filtered, "Segment", "Churn rate by segment", order=SEGMENT_ORDER), key="an_segments")
    with right:
        ui.chart_card(charts.segment_heatmap(filtered), key="an_heatmap")
    table = churn_rate_by(filtered, "Segment")
    st.dataframe(
        table.rename(columns={"customers": "Customers", "churned": "Churned", "churn_rate": "Churn rate",
                              "avg_tenure": "Avg tenure (months)", "avg_monthly_charges": "Avg monthly charges"}),
        hide_index=True, width="stretch",
        column_config={"Churn rate": st.column_config.NumberColumn(format="percent"),
                       "Avg tenure (months)": st.column_config.NumberColumn(format="%.1f"),
                       "Avg monthly charges": st.column_config.NumberColumn(format="%.2f")},
    )
