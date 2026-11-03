"""Plotly figure builders. They take data in and return figures; no Streamlit calls."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.theme import (
    AMBER, BLUE, CORAL, MODEL_COLORS, MUTED, RISK_COLORS, STATUS_COLORS, SURFACE_RAISED, TEAL, TEXT,
    style_figure,
)
from src.eda import churn_rate_by

PERCENT = ".0%"


def churn_donut(df: pd.DataFrame, title: str = "Churn distribution") -> go.Figure:
    counts = df["Status"].value_counts().reindex(["Retained", "Churned"]).fillna(0)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.7, sort=False, textinfo="none",
        marker=dict(colors=[STATUS_COLORS[s] for s in counts.index], line=dict(color=SURFACE_RAISED, width=3)),
        hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>",
    ))
    rate = counts["Churned"] / counts.sum() if counts.sum() else 0
    fig.add_annotation(text=f"<b>{rate:.1%}</b><br><span style='font-size:11px;color:{MUTED}'>churn rate</span>",
                       showarrow=False, font=dict(size=24, color=TEXT))
    return style_figure(fig, height=320, title=title)


def churn_rate_bar(df: pd.DataFrame, column: str, title: str, order: list[str] | None = None,
                   height: int = 320) -> go.Figure:
    """Horizontal bars of churn rate per category. Bars above the overall rate are coral."""
    table = churn_rate_by(df, column)
    table[column] = table[column].astype(str)
    if order:
        table = table.set_index(column).reindex([o for o in order if o in set(table[column])]).reset_index()
    else:
        table = table.sort_values("churn_rate", ascending=False)
    overall = df["ChurnFlag"].mean() if len(df) else 0
    colors = [CORAL if rate > overall else TEAL for rate in table["churn_rate"]]
    fig = go.Figure(go.Bar(
        x=table["churn_rate"], y=table[column], orientation="h", marker_color=colors,
        text=[f"{rate:.1%}" for rate in table["churn_rate"]], textposition="outside", cliponaxis=False,
        customdata=np.stack([table["customers"], table["churned"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Churn rate: %{x:.1%}<br>Customers: %{customdata[0]:,}"
                       "<br>Churned: %{customdata[1]:,}<extra></extra>"),
    ))
    fig.add_vline(x=overall, line_dash="dot", line_color=MUTED,
                  annotation_text=f"Overall {overall:.1%}", annotation_font=dict(size=11, color=MUTED),
                  annotation_position="bottom right")
    fig.update_xaxes(tickformat=PERCENT, range=[0, max(table["churn_rate"].max(), overall) * 1.3 if len(table) else 1],
                     title="Churn rate")
    fig.update_yaxes(autorange="reversed", title="")
    return style_figure(fig, height=height, title=title, show_legend=False)


def distribution_by_status(df: pd.DataFrame, column: str, title: str, x_title: str, bins: int = 30,
                           height: int = 320) -> go.Figure:
    fig = px.histogram(df, x=column, color="Status", nbins=bins, histnorm="percent", barmode="overlay",
                       opacity=0.65, color_discrete_map=STATUS_COLORS,
                       category_orders={"Status": ["Retained", "Churned"]})
    fig.update_traces(hovertemplate=f"{x_title}: %{{x}}<br>%{{y:.1f}}% of group<extra>%{{fullData.name}}</extra>")
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title="% of customers in group")
    return style_figure(fig, height=height, title=title)


def segment_heatmap(df: pd.DataFrame, title: str = "Churn rate by tenure and contract") -> go.Figure:
    grouped = df.groupby(["TenureBand", "Contract"], observed=True).agg(
        rate=("ChurnFlag", "mean"), customers=("ChurnFlag", "size")).reset_index()
    rates = grouped.pivot(index="TenureBand", columns="Contract", values="rate")
    counts = grouped.pivot(index="TenureBand", columns="Contract", values="customers")
    fig = go.Figure(go.Heatmap(
        z=rates.values, x=rates.columns.astype(str), y=rates.index.astype(str), customdata=counts.values,
        colorscale=[[0, "#16324F"], [0.5, "#2C7C86"], [1, CORAL]], zmin=0,
        text=np.where(np.isnan(rates.values), "", np.vectorize(lambda v: f"{v:.0%}")(np.nan_to_num(rates.values))),
        texttemplate="%{text}", hoverongaps=False,
        hovertemplate="<b>%{y}, %{x}</b><br>Churn rate: %{z:.1%}<br>Customers: %{customdata:,}<extra></extra>",
        colorbar=dict(tickformat=PERCENT, thickness=10, outlinewidth=0),
    ))
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, height=320, title=title, show_legend=False)


# ------------------------------------------------------------------ prediction page
def probability_gauge(probability: float, risk: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100,
        number=dict(suffix="%", valueformat=".1f", font=dict(size=44, color=TEXT)),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 35, 50, 65, 100], tickfont=dict(size=11, color=MUTED), tickcolor=MUTED),
            bar=dict(color=RISK_COLORS[risk], thickness=0.3), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 35], color="rgba(52,209,191,0.13)"),
                   dict(range=[35, 65], color="rgba(242,180,90,0.13)"),
                   dict(range=[65, 100], color="rgba(244,114,124,0.13)")],
            threshold=dict(line=dict(color=TEXT, width=2), thickness=0.85, value=50),
        ),
    ))
    fig.update_layout(margin=dict(l=24, r=24, t=24, b=0), height=240, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TEXT))
    return fig


def driver_bar(drivers: pd.DataFrame) -> go.Figure:
    """Top feature contributions for one customer (SHAP, in log-odds)."""
    table = drivers.iloc[::-1]
    labels = [f"{row.feature} = {row.value:g}" if isinstance(row.value, (int, float, np.number))
              else f"{row.feature} = {row.value}" for row in table.itertuples()]
    fig = go.Figure(go.Bar(
        x=table["shap_value"], y=labels, orientation="h",
        marker_color=[CORAL if v > 0 else TEAL for v in table["shap_value"]],
        hovertemplate="<b>%{y}</b><br>Impact: %{x:+.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_xaxes(title="Pushes towards stay  |  towards churn  (log-odds)")
    return style_figure(fig, height=330, title="What drives this prediction", show_legend=False)


# ------------------------------------------------------------------ batch page
def probability_histogram(probabilities: pd.Series, title: str = "Distribution of predicted churn probability") -> go.Figure:
    fig = go.Figure(go.Histogram(x=probabilities, xbins=dict(start=0, end=1, size=0.05), marker_color=TEAL,
                                 hovertemplate="Probability %{x:.0%}<br>%{y:,} customers<extra></extra>"))
    fig.add_vline(x=0.5, line_dash="dot", line_color=MUTED, annotation_text="Threshold 50%",
                  annotation_font=dict(size=11, color=MUTED))
    fig.update_xaxes(tickformat=PERCENT, title="Predicted churn probability", range=[0, 1])
    fig.update_yaxes(title="Customers")
    return style_figure(fig, height=320, title=title, show_legend=False)


def risk_bar(risk_series: pd.Series, title: str = "Customers by risk category") -> go.Figure:
    counts = risk_series.value_counts().reindex(["Low", "Medium", "High"]).fillna(0).astype(int)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color=[RISK_COLORS[r] for r in counts.index],
        text=counts.values, textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x} risk</b><br>%{y:,} customers<extra></extra>",
    ))
    fig.update_yaxes(title="Customers")
    return style_figure(fig, height=320, title=title, show_legend=False)


# ------------------------------------------------------------------ model page
def metric_comparison(comparison: dict) -> go.Figure:
    metrics = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
               ("f1", "F1-score"), ("roc_auc", "ROC-AUC")]
    fig = go.Figure()
    for model, values in comparison.items():
        fig.add_bar(name=model, x=[label for _, label in metrics], y=[values[f"test_{key}"] for key, _ in metrics],
                    marker_color=MODEL_COLORS.get(model, BLUE),
                    hovertemplate="%{x}: %{y:.3f}<extra>" + model + "</extra>")
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 1], tickformat=".1f", title="Score (test set)")
    return style_figure(fig, height=340, title="Test-set metrics by model")


def confusion_matrix_figure(matrix: dict, title: str = "Confusion matrix (test set)") -> go.Figure:
    z = [[matrix["tn"], matrix["fp"]], [matrix["fn"], matrix["tp"]]]
    total = sum(sum(row) for row in z)
    text = [[f"<b>{value:,}</b><br>{value / total:.1%}" for value in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=["Predicted retained", "Predicted churned"], y=["Actually retained", "Actually churned"],
        text=text, texttemplate="%{text}", colorscale=[[0, "#13243F"], [1, "#2A8F87"]], showscale=False,
        hovertemplate="%{y}, %{x}<br>%{z:,} customers<extra></extra>", textfont=dict(size=15),
    ))
    fig.update_yaxes(autorange="reversed")
    return style_figure(fig, height=340, title=title, show_legend=False)


def roc_figure(evaluation: dict, comparison: dict, deployed: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot", width=1),
                    name="Random guess", hoverinfo="skip")
    for model, result in evaluation.items():
        curve = result["roc_curve"]
        auc = comparison[model]["test_roc_auc"]
        fig.add_scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=f"{model} (AUC {auc:.3f})",
                        line=dict(color=MODEL_COLORS.get(model, BLUE), width=3 if model == deployed else 1.6),
                        hovertemplate="False positive rate %{x:.2f}<br>True positive rate %{y:.2f}<extra></extra>")
    fig.update_xaxes(title="False positive rate", range=[0, 1])
    fig.update_yaxes(title="True positive rate", range=[0, 1.02])
    return style_figure(fig, height=340, title="ROC curves (test set)")


def importance_bar(records: list[dict], value_key: str, title: str, color: str = TEAL, top_n: int = 12) -> go.Figure:
    table = pd.DataFrame(records).sort_values(value_key, ascending=False).head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(x=table[value_key], y=table["feature"], orientation="h", marker_color=color,
                           hovertemplate="<b>%{y}</b><br>%{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Importance")
    return style_figure(fig, height=380, title=title, show_legend=False)
