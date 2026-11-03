"""Cached loaders. Streamlit reruns the script on every interaction, so anything
expensive (reading data, loading the model) is cached here and never re-trained."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import METRICS_PATH, MODEL_PATH
from src.data_loader import is_synthetic_dataset, load_clean_dataset, resolve_dataset_path
from src.eda import add_derived_columns
from src.explainability import build_explainer
from src.predict import ModelNotFoundError, load_pipeline


def _modified_time(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading dataset...")
def _load_customer_data(cache_key: float) -> pd.DataFrame:
    return add_derived_columns(load_clean_dataset())


def load_customer_data() -> pd.DataFrame:
    """Cleaned dataset with derived columns; stops the page with a clear message if missing."""
    path = resolve_dataset_path()
    if path is None:
        st.error("No dataset found. Run `python -m src.data_loader` from the project folder, then refresh.")
        st.stop()
    try:
        return _load_customer_data(_modified_time(path))
    except (ValueError, FileNotFoundError) as error:
        st.error(f"The dataset could not be loaded: {error}")
        st.stop()


def data_source_pills() -> list[tuple[str, str]]:
    if is_synthetic_dataset():
        return [("Synthetic dataset, not real customers", "amber")]
    return [("IBM Telco sample dataset", "teal")]


@st.cache_resource(show_spinner="Loading model...")
def _load_pipeline(cache_key: float):
    return load_pipeline()


def get_pipeline():
    """Saved pipeline, loaded once. Stops the page with a helpful message if it is missing."""
    try:
        return _load_pipeline(_modified_time(MODEL_PATH))
    except ModelNotFoundError as error:
        st.error(str(error))
        st.stop()


@st.cache_resource(show_spinner=False)
def _build_explainer(cache_key: float):
    return build_explainer(_load_pipeline(cache_key))


def get_explainer():
    get_pipeline()  # makes sure the model exists first
    return _build_explainer(_modified_time(MODEL_PATH))


@st.cache_data(show_spinner=False)
def _read_metrics(cache_key: float) -> dict | None:
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def load_metrics() -> dict | None:
    return _read_metrics(_modified_time(METRICS_PATH))


@st.cache_data(show_spinner=False)
def sample_template_csv(cache_key: float) -> bytes:
    """A small CSV in the exact upload format, taken from the loaded dataset (no target column)."""
    from src.config import FEATURE_COLUMNS, ID_COLUMN
    df = _load_customer_data(cache_key)
    sample = df.sample(20, random_state=7)[[ID_COLUMN] + FEATURE_COLUMNS]
    return sample.to_csv(index=False).encode("utf-8")


def get_sample_template() -> bytes:
    path = resolve_dataset_path()
    load_customer_data()
    return sample_template_csv(_modified_time(path))
