
"""Smoke tests for the Streamlit app using Streamlit's own test harness."""
from pathlib import Path
import sys

import pytest
from streamlit.testing.v1 import AppTest

from src.config import MODEL_PATH

APP_DIR = Path(__file__).resolve().parents[1] / "app"

# Make app/components importable when AppTest runs pages directly.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

PAGES = [
    "executive_dashboard",
    "customer_prediction",
    "batch_prediction",
    "customer_analytics",
    "model_performance",
    "about",
]

pytestmark = pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="run `python -m src.train` first",
)


def run_page(name: str) -> AppTest:
    return AppTest.from_file(
        str(APP_DIR / "pages" / f"{name}.py"),
        default_timeout=120,
    ).run()


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_errors(page):
    app = run_page(page)
    assert not app.exception, app.exception
    assert not app.error


def test_full_app_with_navigation_starts():
    app = AppTest.from_file(
        str(APP_DIR / "app.py"),
        default_timeout=120,
    ).run()
    assert not app.exception
    assert any("Executive Dashboard" in item.value for item in app.markdown)


def test_prediction_button_produces_a_result():
    app = run_page("customer_prediction")
    app.button[0].click().run()
    assert not app.exception
    assert any("riq-verdict-label" in item.value for item in app.markdown)


def test_no_internet_locks_add_on_fields():
    app = run_page("customer_prediction")
    app.selectbox(key="in_internet").set_value("No").run()
    assert not app.exception
    locked = [
        field for field in app.text_input
        if field.value == "No internet service"
    ]
    assert len(locked) == 6


def test_analytics_filters_with_no_matches_show_a_message():
    app = run_page("customer_analytics")
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any(
        "No customers match" in warning.value
        for warning in app.warning
    )


@pytest.fixture()
def metrics_temporarily_missing(tmp_path):
    """Move the real metrics file away for one test, then always put it back."""
    from src.config import METRICS_PATH

    backup = tmp_path / "metrics_backup.json"
    METRICS_PATH.replace(backup)
    try:
        yield
    finally:
        backup.replace(METRICS_PATH)


def test_model_page_reports_missing_metrics(metrics_temporarily_missing):
    app = run_page("model_performance")
    assert not app.exception
    assert any(
        "Train the model first" in warning.value
        for warning in app.warning
    )


# ---------------------------------------------------------------- batch upload flow

def upload_csv(csv_text: str) -> AppTest:
    app = run_page("batch_prediction")
    app.file_uploader[0].set_value(
        ("customers.csv", csv_text.encode("utf-8"), "text/csv")
    ).run()
    return app


def test_batch_upload_end_to_end(clean_df):
    from src.config import FEATURE_COLUMNS

    csv_text = clean_df.head(40)[
        ["customerID"] + FEATURE_COLUMNS
    ].to_csv(index=False)

    app = upload_csv(csv_text)
    assert not app.exception
    assert any("Validation passed" in item.value for item in app.success)

    app.button[0].click().run()
    assert not app.exception
    assert any("Customers scored" in item.value for item in app.markdown)


def test_batch_upload_with_missing_column_shows_error(clean_df):
    from src.config import FEATURE_COLUMNS

    csv_text = clean_df.head(10)[FEATURE_COLUMNS].drop(
        columns=["Contract"]
    ).to_csv(index=False)

    app = upload_csv(csv_text)
    assert not app.exception
    assert any(
        "Missing required column" in item.value
        for item in app.error
    )


def test_batch_upload_with_invalid_value_shows_error(clean_df):
    from src.config import FEATURE_COLUMNS

    broken = clean_df.head(10)[FEATURE_COLUMNS].copy()
    broken.loc[broken.index[0], "PaymentMethod"] = "Bitcoin"

    app = upload_csv(broken.to_csv(index=False))
    assert any("Bitcoin" in item.value for item in app.error)


def test_batch_upload_of_empty_file_shows_error():
    app = upload_csv("")
    assert not app.exception
    assert any("empty" in item.value.lower() for item in app.error)
