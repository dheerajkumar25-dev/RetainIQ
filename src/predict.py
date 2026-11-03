"""Inference: load the saved pipeline and turn customer data into churn predictions.

No Streamlit code here, so the same functions work from a script, a test or the app.
"""
import joblib
import pandas as pd

from src.config import (
    CLASSIFICATION_THRESHOLD,
    MODEL_PATH,
    RISK_HIGH_LOWER,
    RISK_LOW_UPPER,
)
from src.validation import InvalidInputError, validate_customer_frame, validate_single_customer

LIKELY_TO_CHURN = "Likely to churn"
LIKELY_TO_STAY = "Likely to stay"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file is missing or cannot be read."""


def load_pipeline(path=MODEL_PATH):
    """Load the saved preprocessing + model pipeline."""
    if not path.exists():
        raise ModelNotFoundError(f"Trained model not found at {path}. Run `python -m src.train` first.")
    try:
        return joblib.load(path)
    except Exception as error:  # usually a scikit-learn / xgboost version mismatch
        raise ModelNotFoundError(
            f"The saved model could not be loaded ({error}). It was probably created with different "
            "library versions. Re-train it with `python -m src.train`."
        ) from error


def get_risk_category(probability: float) -> str:
    if probability < RISK_LOW_UPPER:
        return "Low"
    if probability < RISK_HIGH_LOWER:
        return "Medium"
    return "High"


def predict_probabilities(pipeline, features: pd.DataFrame):
    return pipeline.predict_proba(features)[:, 1]


def predict_batch(pipeline, customers: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    """Return a table with churn probability, predicted class and risk for each customer.

    `customers` may contain extra columns (e.g. customerID); they are kept in the output.
    Pass validate=False only if the data was already validated with `validate_customer_frame`.
    """
    if validate:
        result = validate_customer_frame(customers)
        if not result.is_valid:
            raise InvalidInputError(result.errors)
        features = result.data
    else:
        features = customers

    probabilities = predict_probabilities(pipeline, features)
    output = customers.copy()
    output["churn_probability"] = probabilities.round(4)
    output["predicted_class"] = [LIKELY_TO_CHURN if p >= CLASSIFICATION_THRESHOLD else LIKELY_TO_STAY
                                 for p in probabilities]
    output["risk_category"] = [get_risk_category(p) for p in probabilities]
    return output


def predict_single(pipeline, customer: dict) -> dict:
    """Predict churn for one customer given as {column: value}. Raises InvalidInputError if invalid."""
    validation = validate_single_customer(customer)
    if not validation.is_valid:
        raise InvalidInputError(validation.errors)
    probability = float(predict_probabilities(pipeline, validation.data)[0])
    return {
        "churn_probability": probability,
        "predicted_class": LIKELY_TO_CHURN if probability >= CLASSIFICATION_THRESHOLD else LIKELY_TO_STAY,
        "risk_category": get_risk_category(probability),
        "warnings": validation.warnings,
        "features": validation.data,  # cleaned one-row frame, handy for explanations
    }
