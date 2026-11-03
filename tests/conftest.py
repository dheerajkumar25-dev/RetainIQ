"""Shared fixtures. Tests use a small sample of the real dataset so they stay fast."""
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import FEATURE_COLUMNS
from src.data_loader import load_clean_dataset
from src.preprocessing import build_preprocessor, make_train_test_split


@pytest.fixture(scope="session")
def clean_df():
    return load_clean_dataset()


@pytest.fixture(scope="session")
def sample_df(clean_df):
    return clean_df.sample(1200, random_state=0).reset_index(drop=True)


@pytest.fixture(scope="session")
def small_pipeline(sample_df):
    """A quickly trained pipeline (not the saved production model)."""
    X_train, _, y_train, _ = make_train_test_split(sample_df)
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", XGBClassifier(n_estimators=40, max_depth=3, eval_metric="logloss", random_state=42)),
    ])
    return pipeline.fit(X_train, y_train)


@pytest.fixture()
def valid_customer(clean_df):
    return clean_df.iloc[1][FEATURE_COLUMNS].to_dict()
