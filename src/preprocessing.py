"""Preprocessing: the sklearn transformer, the train/test split and small helpers.

The transformer is *only* defined here. It is fitted inside the training
pipeline on the training split, and the very same fitted object is saved with
the model, so inference always applies identical transformations.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical columns.

    Scaling matters for Logistic Regression and is harmless for tree models, so
    one shared preprocessor keeps the comparison between models fair.
    Unknown categories at inference time are encoded as all zeros instead of crashing.
    """
    numeric_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_steps, NUMERIC_FEATURES),
        ("cat", categorical_steps, CATEGORICAL_FEATURES),
    ])


def encode_target(churn_labels: pd.Series) -> pd.Series:
    """Convert 'Yes'/'No' churn labels to 1/0."""
    encoded = churn_labels.map({"Yes": 1, "No": 0})
    if encoded.isna().any():
        raise ValueError("Churn column contains values other than 'Yes' / 'No'.")
    return encoded.astype(int)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return the model input columns and the 0/1 target."""
    return df[FEATURE_COLUMNS].copy(), encode_target(df[TARGET_COLUMN])


def make_train_test_split(df: pd.DataFrame, test_size: float = TEST_SIZE,
                          random_state: int = RANDOM_STATE):
    """Stratified split, so train and test keep the same churn rate."""
    features, target = split_features_target(df)
    return train_test_split(features, target, test_size=test_size,
                            random_state=random_state, stratify=target)


def normalize_senior_citizen(values: pd.Series) -> pd.Series:
    """Accept 0/1, 'Yes'/'No' (any case) and return 'No'/'Yes'. Unknown values become NaN."""
    mapping = {"0": "No", "0.0": "No", "no": "No", "1": "Yes", "1.0": "Yes", "yes": "Yes"}
    as_text = values.astype("string").str.strip().str.lower()
    return as_text.map(mapping).astype(object)


def map_encoded_to_original(encoded_names) -> list[str]:
    """Map names like 'cat__Contract_One year' back to the original column 'Contract'."""
    original = []
    for name in encoded_names:
        stripped = name.split("__", 1)[1] if "__" in name else name
        if stripped in NUMERIC_FEATURES:
            original.append(stripped)
            continue
        matches = [col for col in CATEGORICAL_FEATURES if stripped.startswith(col + "_")]
        if not matches:
            raise ValueError(f"Cannot map encoded feature '{name}' to an original column.")
        original.append(max(matches, key=len))
    return original
