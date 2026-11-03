import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, MODEL_PATH
from src.predict import (
    LIKELY_TO_CHURN, LIKELY_TO_STAY, ModelNotFoundError, get_risk_category, load_pipeline, predict_batch,
    predict_single,
)
from src.validation import InvalidInputError


def test_risk_categories_at_boundaries():
    assert get_risk_category(0.0) == "Low"
    assert get_risk_category(0.349) == "Low"
    assert get_risk_category(0.35) == "Medium"
    assert get_risk_category(0.649) == "Medium"
    assert get_risk_category(0.65) == "High"
    assert get_risk_category(1.0) == "High"


def test_single_prediction_structure(small_pipeline, valid_customer):
    result = predict_single(small_pipeline, valid_customer)
    assert 0 <= result["churn_probability"] <= 1
    assert result["predicted_class"] in {LIKELY_TO_CHURN, LIKELY_TO_STAY}
    assert result["risk_category"] in {"Low", "Medium", "High"}
    assert list(result["features"].columns) == FEATURE_COLUMNS


def test_high_risk_profile_scores_higher_than_loyal_profile(small_pipeline, valid_customer):
    risky = dict(valid_customer, Contract="Month-to-month", tenure=1, MonthlyCharges=95.0, TotalCharges=95.0,
                 InternetService="Fiber optic", PaymentMethod="Electronic check", TechSupport="No", OnlineSecurity="No")
    loyal = dict(valid_customer, Contract="Two year", tenure=70, MonthlyCharges=60.0, TotalCharges=4200.0,
                 InternetService="DSL", PaymentMethod="Credit card (automatic)", TechSupport="Yes", OnlineSecurity="Yes")
    assert predict_single(small_pipeline, risky)["churn_probability"] > predict_single(small_pipeline, loyal)["churn_probability"]


@pytest.mark.parametrize("field, bad_value", [
    ("Contract", "Weekly"), ("tenure", -3), ("tenure", "abc"), ("MonthlyCharges", 99999), ("PaymentMethod", None),
])
def test_invalid_single_inputs_raise(small_pipeline, valid_customer, field, bad_value):
    with pytest.raises(InvalidInputError):
        predict_single(small_pipeline, dict(valid_customer, **{field: bad_value}))


def test_missing_field_raises(small_pipeline, valid_customer):
    del valid_customer["tenure"]
    with pytest.raises(InvalidInputError):
        predict_single(small_pipeline, valid_customer)


def test_batch_prediction_keeps_extra_columns(small_pipeline, clean_df):
    batch = clean_df.head(30)
    result = predict_batch(small_pipeline, batch)
    assert len(result) == 30
    assert {"customerID", "churn_probability", "predicted_class", "risk_category"} <= set(result.columns)
    assert result["churn_probability"].between(0, 1).all()


def test_batch_prediction_handles_missing_values(small_pipeline, clean_df):
    batch = clean_df.head(10).copy()
    batch.loc[batch.index[0], "Contract"] = None
    batch.loc[batch.index[1], "tenure"] = None
    assert predict_batch(small_pipeline, batch)["churn_probability"].notna().all()


def test_batch_prediction_rejects_bad_files(small_pipeline, clean_df):
    with pytest.raises(InvalidInputError):
        predict_batch(small_pipeline, clean_df.head(5).drop(columns=["Contract"]))
    with pytest.raises(InvalidInputError):
        predict_batch(small_pipeline, pd.DataFrame())


def test_batch_matches_single_prediction(small_pipeline, clean_df):
    row = clean_df.iloc[3]
    single = predict_single(small_pipeline, row[FEATURE_COLUMNS].to_dict())["churn_probability"]
    batch = predict_batch(small_pipeline, clean_df.iloc[[3]])["churn_probability"].iloc[0]
    assert single == pytest.approx(batch, abs=1e-4)


def test_missing_model_file_gives_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python -m src.train"):
        load_pipeline(tmp_path / "missing.joblib")


def test_corrupt_model_file_gives_clear_error(tmp_path):
    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"not a model")
    with pytest.raises(ModelNotFoundError):
        load_pipeline(broken)


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="run `python -m src.train` first")
def test_saved_production_model_predicts(clean_df, valid_customer):
    pipeline = load_pipeline()
    assert 0 <= predict_single(pipeline, valid_customer)["churn_probability"] <= 1
    assert len(predict_batch(pipeline, clean_df.head(20))) == 20
