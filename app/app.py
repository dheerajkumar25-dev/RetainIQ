"""RetainIQ entry point.  Run with:  streamlit run app/app.py"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
for folder in (ROOT_DIR, APP_DIR):  # so `src` and `components` import no matter where Streamlit is started
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="RetainIQ", page_icon=":material/insights:", layout="wide",
                   initial_sidebar_state="expanded")

from components.data_access import load_metrics  # noqa: E402
from components.theme import apply_theme  # noqa: E402
from src.data_loader import is_synthetic_dataset  # noqa: E402

apply_theme()
st.logo(str(APP_DIR / "assets" / "logo.svg"), icon_image=str(APP_DIR / "assets" / "icon.svg"), size="large")

PAGES_DIR = APP_DIR / "pages"
navigation = st.navigation([
    st.Page(PAGES_DIR / "executive_dashboard.py", title="Executive Dashboard", icon=":material/space_dashboard:",
            url_path="dashboard", default=True),
    st.Page(PAGES_DIR / "customer_prediction.py", title="Customer Prediction", icon=":material/person_search:",
            url_path="prediction"),
    st.Page(PAGES_DIR / "batch_prediction.py", title="Batch Prediction", icon=":material/upload_file:",
            url_path="batch"),
    st.Page(PAGES_DIR / "customer_analytics.py", title="Customer Analytics", icon=":material/bar_chart:",
            url_path="analytics"),
    st.Page(PAGES_DIR / "model_performance.py", title="Model Performance", icon=":material/model_training:",
            url_path="model"),
    st.Page(PAGES_DIR / "about.py", title="About", icon=":material/info:", url_path="about"),
])

metrics = load_metrics()
model_line = (f"<b>Model</b><br>{metrics['deployed_model']}, trained {metrics['trained_at_utc'][:10]}"
              if metrics else "<b>Model</b><br>not trained yet")
data_line = "<b>Data</b><br>" + ("Synthetic (not real customers)" if is_synthetic_dataset() else "IBM Telco sample dataset")
with st.sidebar:
    st.markdown(f'<div class="riq-sidebar-note">{model_line}<br><br>{data_line}<br><br>'
                "Predict churn. Understand customers. Improve retention.</div>", unsafe_allow_html=True)

navigation.run()
