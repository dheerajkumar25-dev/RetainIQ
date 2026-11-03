"""Data loading, cleaning and quality checks.

Run `python -m src.data_loader` to prepare the dataset. It uses the IBM Telco
file in data/raw/ if present, tries to download it if not, and only falls back
to a clearly labelled synthetic dataset when neither is possible.
"""
import json
import urllib.request

import numpy as np
import pandas as pd

from src.config import (
    CATEGORY_OPTIONS,
    CLEAN_DATASET_PATH,
    DATA_PROCESSED_DIR,
    DATASET_INFO_PATH,
    DATASET_URL,
    ID_COLUMN,
    RAW_DATASET_PATH,
    REPORTS_DIR,
    SYNTHETIC_DATASET_PATH,
    TARGET_COLUMN,
)
from src.preprocessing import normalize_senior_citizen

REQUIRED_RAW_COLUMNS = [ID_COLUMN, "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "Contract", TARGET_COLUMN]


# ---------------------------------------------------------------- loading
def load_raw_dataset(path=None) -> pd.DataFrame:
    """Read a raw churn CSV and check that the key columns exist."""
    path = path or resolve_dataset_path()
    if path is None or not path.exists():
        raise FileNotFoundError(
            "No dataset found in data/raw/. Run `python -m src.data_loader` to download or generate one."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    missing = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {missing}")
    return df


def resolve_dataset_path():
    """Prefer the real IBM file; use the synthetic file only if that is all we have."""
    if RAW_DATASET_PATH.exists():
        return RAW_DATASET_PATH
    if SYNTHETIC_DATASET_PATH.exists():
        return SYNTHETIC_DATASET_PATH
    return None


def is_synthetic_dataset(path=None) -> bool:
    path = path or resolve_dataset_path()
    return path == SYNTHETIC_DATASET_PATH


def download_raw_dataset(url: str = DATASET_URL, destination=RAW_DATASET_PATH, timeout: int = 30) -> bool:
    """Download the IBM Telco CSV. Returns False (instead of raising) if it is unavailable."""
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=timeout) as response:
            temp_path.write_bytes(response.read())
        load_raw_dataset(temp_path)  # sanity-check before keeping it
        temp_path.replace(destination)
        return True
    except Exception as error:  # network problems, bad file, etc.
        print(f"Download failed: {error}")
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()
        return False


# ---------------------------------------------------------------- cleaning
def clean_dataset(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data (no row is dropped silently)."""
    df = raw_df.copy()
    df.columns = df.columns.str.strip()

    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].str.strip()

    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # In the source data TotalCharges is blank only for customers with tenure 0.
    # They have not been billed yet, so 0 is the correct value.
    brand_new = df["TotalCharges"].isna() & (df["tenure"] == 0)
    df.loc[brand_new, "TotalCharges"] = 0.0

    df["SeniorCitizen"] = normalize_senior_citizen(df["SeniorCitizen"])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(object)
    return df


def load_clean_dataset() -> pd.DataFrame:
    return clean_dataset(load_raw_dataset())


# ---------------------------------------------------------------- quality report
def data_quality_report(df: pd.DataFrame) -> dict:
    """Missing values, duplicates, dtypes and target balance as a plain dict."""
    missing = df.isna().sum()
    churn_counts = df[TARGET_COLUMN].value_counts()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_values": {col: int(n) for col, n in missing.items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df[ID_COLUMN].duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "churn_counts": {str(k): int(v) for k, v in churn_counts.items()},
        "churn_rate": float((df[TARGET_COLUMN] == "Yes").mean()),
    }


def find_unexpected_categories(df: pd.DataFrame) -> dict:
    """Categories in the data that the form/validation do not know about (should be empty)."""
    unexpected = {}
    for column, allowed in CATEGORY_OPTIONS.items():
        extra = sorted(set(df[column].dropna().unique()) - set(allowed))
        if extra:
            unexpected[column] = extra
    return unexpected


# ---------------------------------------------------------------- synthetic fallback
def generate_synthetic_dataset(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Reproducible SYNTHETIC churn data with the same columns as the IBM file.

    Only used if the real dataset cannot be obtained. The churn label is drawn from
    a hand-written logistic formula, so it contains no real customer behaviour.
    """
    rng = np.random.default_rng(seed)
    n = n_customers

    contract = rng.choice(CATEGORY_OPTIONS["Contract"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(CATEGORY_OPTIONS["InternetService"], n, p=[0.34, 0.44, 0.22])
    payment = rng.choice(CATEGORY_OPTIONS["PaymentMethod"], n, p=[0.34, 0.23, 0.22, 0.21])
    tenure = np.clip(rng.gamma(shape=1.6, scale=22, size=n).astype(int), 0, 72)
    tenure = np.where(contract == "Two year", np.maximum(tenure, 12), tenure)

    def yes_no(prob):
        return rng.choice(["Yes", "No"], n, p=[prob, 1 - prob])

    phone = yes_no(0.9)
    multiple_lines = np.where(phone == "No", "No phone service", yes_no(0.42))
    add_ons = {}
    for name, prob in [("OnlineSecurity", 0.3), ("OnlineBackup", 0.35), ("DeviceProtection", 0.34),
                       ("TechSupport", 0.3), ("StreamingTV", 0.4), ("StreamingMovies", 0.4)]:
        add_ons[name] = np.where(internet == "No", "No internet service", yes_no(prob))

    base_price = np.select([internet == "DSL", internet == "Fiber optic"], [45, 75], default=22)
    n_add_ons = sum((values == "Yes").astype(int) for values in add_ons.values())
    monthly = np.round(base_price + 5 * n_add_ons + rng.normal(0, 4, n), 2).clip(18, 120)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.1, n), 2)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])

    logit = (-0.8 + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
             + 0.6 * (internet == "Fiber optic") - 0.03 * tenure + 0.008 * (monthly - 65)
             + 0.45 * (payment == "Electronic check") - 0.4 * (add_ons["TechSupport"] == "Yes")
             - 0.3 * (add_ons["OnlineSecurity"] == "Yes") + 0.25 * senior)
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame({
        ID_COLUMN: [f"SYN-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": senior,
        "Partner": yes_no(0.48),
        "Dependents": yes_no(0.3),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        **add_ons,
        "Contract": contract,
        "PaperlessBilling": yes_no(0.59),
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        TARGET_COLUMN: np.where(churn, "Yes", "No"),
    })


# ---------------------------------------------------------------- orchestration
def prepare_dataset() -> tuple[pd.DataFrame, dict]:
    """Make sure a dataset exists, clean it, save a processed copy and a data summary."""
    if resolve_dataset_path() is None:
        print("Dataset not found locally, trying to download the IBM Telco dataset...")
        if not download_raw_dataset():
            print("Download unavailable: generating a SYNTHETIC dataset instead.")
            generate_synthetic_dataset().to_csv(SYNTHETIC_DATASET_PATH, index=False)

    path = resolve_dataset_path()
    synthetic = is_synthetic_dataset(path)
    clean_df = clean_dataset(load_raw_dataset(path))

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEAN_DATASET_PATH, index=False)

    info = {
        "source_file": path.name,
        "is_synthetic": synthetic,
        "source_description": (
            "Synthetic dataset generated by src/data_loader.py (NOT real customer data)."
            if synthetic else
            "IBM Telco Customer Churn sample dataset, publicly released by IBM for demonstrating analytics tools."
        ),
        "quality_report": data_quality_report(clean_df),
        "unexpected_categories": find_unexpected_categories(clean_df),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_INFO_PATH.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return clean_df, info


def write_data_dictionary(path=None) -> None:
    """Generate docs/data_dictionary.md from the single source of truth in config.py."""
    from src.config import DOCS_DIR
    path = path or DOCS_DIR / "data_dictionary.md"
    lines = ["# Data dictionary", "",
             "Generated from `src/config.py` by `python -m src.data_loader`.", "",
             "| Column | Type | Role | Description | Allowed values |", "|---|---|---|---|---|"]
    from src.config import DATA_DICTIONARY
    for column, (kind, role, description) in DATA_DICTIONARY.items():
        allowed = ", ".join(CATEGORY_OPTIONS.get(column, [])) or ("any number" if kind == "numeric" else "")
        if column == "Churn":
            allowed = "Yes, No"
        lines.append(f"| `{column}` | {kind} | {role} | {description} | {allowed} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df, info = prepare_dataset()
    write_data_dictionary()
    report = info["quality_report"]
    print(f"Source: {info['source_description']}")
    print(f"Rows: {report['rows']:,}  Columns: {report['columns']}")
    print(f"Missing values (after cleaning): {report['missing_values'] or 'none'}")
    print(f"Duplicate rows: {report['duplicate_rows']}  Duplicate IDs: {report['duplicate_customer_ids']}")
    print(f"Churn distribution: {report['churn_counts']}  (churn rate {report['churn_rate']:.1%})")
    if info["unexpected_categories"]:
        print(f"WARNING unexpected categories: {info['unexpected_categories']}")
    print(f"Saved: {CLEAN_DATASET_PATH}")


if __name__ == "__main__":
    main()
