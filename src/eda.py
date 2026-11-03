"""Exploratory data analysis: derived columns, segment summaries, insights and static figures.

Run `python -m src.eda` to save the figures in reports/figures and a summary JSON.
The Streamlit pages reuse the same functions, so numbers match everywhere.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import EDA_SUMMARY_PATH, FIGURES_DIR, ID_COLUMN, NUMERIC_FEATURES, TARGET_COLUMN

TENURE_LABELS = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
SEGMENT_ORDER = ["New, month-to-month", "Established, month-to-month", "One-year contract", "Two-year contract"]


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable helper columns used by charts: status, tenure band, charge tier, segment."""
    df = df.copy()
    df["ChurnFlag"] = (df[TARGET_COLUMN] == "Yes").astype(int)
    df["Status"] = np.where(df["ChurnFlag"] == 1, "Churned", "Retained")
    df["TenureBand"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, np.inf], labels=TENURE_LABELS)
    tiers = pd.qcut(df["MonthlyCharges"], q=3, labels=["Low", "Medium", "High"], duplicates="drop")
    df["ChargeTier"] = tiers
    df["Segment"] = assign_segment(df)
    return df


def assign_segment(df: pd.DataFrame) -> pd.Series:
    """Simple, explainable business segments based on contract and tenure."""
    segment = np.select(
        [
            (df["Contract"] == "Month-to-month") & (df["tenure"] <= 12),
            df["Contract"] == "Month-to-month",
            df["Contract"] == "One year",
        ],
        SEGMENT_ORDER[:3],
        default=SEGMENT_ORDER[3],
    )
    return pd.Series(pd.Categorical(segment, categories=SEGMENT_ORDER, ordered=True), index=df.index)


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Customers, churned customers and churn rate for each value of `column`."""
    table = (df.groupby(column, observed=True)
             .agg(customers=("ChurnFlag", "size"), churned=("ChurnFlag", "sum"), churn_rate=("ChurnFlag", "mean"),
                  avg_tenure=("tenure", "mean"), avg_monthly_charges=("MonthlyCharges", "mean"))
             .reset_index())
    return table


def overall_summary(df: pd.DataFrame) -> dict:
    churned = int(df["ChurnFlag"].sum())
    return {
        "total_customers": int(len(df)),
        "churned": churned,
        "retained": int(len(df) - churned),
        "churn_rate": float(df["ChurnFlag"].mean()),
        "avg_monthly_charges": float(df["MonthlyCharges"].mean()),
        "avg_tenure": float(df["tenure"].mean()),
    }


def generate_insights(df: pd.DataFrame) -> list[str]:
    """Plain-language observations computed from the data (nothing hardcoded)."""
    insights = []
    overall = overall_summary(df)
    insights.append(f"{overall['churned']:,} of {overall['total_customers']:,} customers "
                    f"({overall['churn_rate']:.1%}) churned in this dataset.")

    def extremes(column, label):
        table = churn_rate_by(df, column).sort_values("churn_rate")
        low, high = table.iloc[0], table.iloc[-1]
        return (f"{label}: churn is highest for '{high[column]}' ({high['churn_rate']:.1%}) and lowest for "
                f"'{low[column]}' ({low['churn_rate']:.1%}).")

    insights.append(extremes("Contract", "Contract type"))
    insights.append(extremes("TenureBand", "Tenure"))
    insights.append(extremes("InternetService", "Internet service"))
    insights.append(extremes("PaymentMethod", "Payment method"))
    by_status = df.groupby("Status")["MonthlyCharges"].mean()
    if {"Churned", "Retained"} <= set(by_status.index):
        insights.append(f"Churned customers paid {by_status['Churned']:.2f} per month on average versus "
                        f"{by_status['Retained']:.2f} for retained customers.")
    return insights


def correlation_with_churn(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df[NUMERIC_FEATURES + ["ChurnFlag"]]
    return numeric.corr()


# ---------------------------------------------------------------- static figures (reports/figures)
def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})


def save_eda_figures(df: pd.DataFrame, output_dir=FIGURES_DIR) -> list[str]:
    _style()
    output_dir.mkdir(parents=True, exist_ok=True)
    palette = {"Retained": "#2BB7A5", "Churned": "#E8636F"}
    saved = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output_dir / name, bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="Status", hue="Status", palette=palette, order=["Retained", "Churned"], legend=False, ax=ax)
    ax.set(title="Churn distribution", xlabel="", ylabel="Customers")
    save(fig, "eda_churn_distribution.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column in zip(axes, NUMERIC_FEATURES):
        sns.histplot(data=df, x=column, hue="Status", palette=palette, bins=30, element="step", ax=ax)
        ax.set_title(f"{column} by churn status")
    save(fig, "eda_numeric_distributions.png")

    for column, name in [("Contract", "eda_churn_by_contract.png"), ("PaymentMethod", "eda_churn_by_payment.png")]:
        table = churn_rate_by(df, column).sort_values("churn_rate", ascending=False)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        sns.barplot(data=table, y=column, x="churn_rate", color="#2BB7A5", ax=ax)
        ax.set(title=f"Churn rate by {column}", xlabel="Churn rate", ylabel="")
        ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        save(fig, name)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.boxplot(data=df, x="Status", y="MonthlyCharges", hue="Status", palette=palette,
                order=["Retained", "Churned"], legend=False, ax=ax)
    ax.set(title="Monthly charges vs churn", xlabel="")
    save(fig, "eda_monthly_charges_vs_churn.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    table = churn_rate_by(df, "TenureBand")
    sns.barplot(data=table, x="TenureBand", y="churn_rate", color="#2BB7A5", ax=ax)
    ax.set(title="Churn rate by tenure band", xlabel="", ylabel="Churn rate")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    save(fig, "eda_tenure_vs_churn.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(correlation_with_churn(df), annot=True, fmt=".2f", cmap="crest", ax=ax)
    ax.set_title("Correlation (numeric features and churn)")
    save(fig, "eda_correlation.png")
    return saved


def main() -> None:
    from src.data_loader import load_clean_dataset
    df = add_derived_columns(load_clean_dataset())
    figures = save_eda_figures(df)
    summary = {"overall": overall_summary(df), "insights": generate_insights(df),
               "churn_rate_by_contract": churn_rate_by(df, "Contract").round(4).to_dict(orient="records")}
    EDA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Saved {len(figures)} figures to {FIGURES_DIR}")
    print("\n".join(f"- {line}" for line in summary["insights"]))


if __name__ == "__main__":
    main()
