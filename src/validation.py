"""Input validation shared by the single-customer form and the CSV batch upload."""
from dataclasses import dataclass, field

import pandas as pd

from src.config import (
    CATEGORICAL_FEATURES,
    CATEGORY_OPTIONS,
    FEATURE_COLUMNS,
    MAX_BATCH_ROWS,
    NUMERIC_FEATURES,
    NUMERIC_HARD_LIMITS,
    NUMERIC_TYPICAL_RANGE,
)
from src.preprocessing import normalize_senior_citizen


class InvalidInputError(ValueError):
    """Raised when customer data cannot be sent to the model."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data: pd.DataFrame | None = None  # cleaned model-ready features (only if valid)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _preview(values, limit: int = 5) -> str:
    shown = ", ".join(str(v) for v in list(values)[:limit])
    return shown + (", ..." if len(values) > limit else "")


def _check_categorical(column: pd.Series, name: str, result: ValidationResult) -> pd.Series:
    allowed = CATEGORY_OPTIONS[name]
    if name == "SeniorCitizen":
        cleaned = normalize_senior_citizen(column)
        invalid = column.notna() & cleaned.isna()
    else:
        canonical = {option.lower(): option for option in allowed}
        cleaned = column.astype("string").str.strip().str.lower().map(canonical).astype(object)
        invalid = column.notna() & cleaned.isna()
    if invalid.any():
        bad_values = column[invalid].astype(str).unique()
        result.errors.append(
            f"'{name}' has {int(invalid.sum())} unrecognised value(s): {_preview(bad_values)}. "
            f"Allowed: {', '.join(allowed)}."
        )
    return cleaned


def _check_numeric(column: pd.Series, name: str, result: ValidationResult) -> pd.Series:
    numbers = pd.to_numeric(column, errors="coerce")
    not_a_number = column.notna() & numbers.isna()
    if not_a_number.any():
        bad_values = column[not_a_number].astype(str).unique()
        result.errors.append(f"'{name}' has {int(not_a_number.sum())} non-numeric value(s): {_preview(bad_values)}.")
        return numbers
    low, high = NUMERIC_HARD_LIMITS[name]
    out_of_bounds = numbers.notna() & ((numbers < low) | (numbers > high))
    if out_of_bounds.any():
        result.errors.append(f"'{name}' has {int(out_of_bounds.sum())} value(s) outside the valid range {low} to {high}.")
        return numbers
    typical_low, typical_high = NUMERIC_TYPICAL_RANGE[name]
    unusual = numbers.notna() & ((numbers < typical_low) | (numbers > typical_high))
    if unusual.any():
        result.warnings.append(
            f"'{name}' has {int(unusual.sum())} value(s) outside the range the model was trained on "
            f"({typical_low} to {typical_high}); predictions for these rows are less reliable."
        )
    return numbers


def _consistency_warnings(features: pd.DataFrame, result: ValidationResult) -> None:
    """Soft checks for combinations that look wrong but are not strictly invalid."""
    tenure, monthly, total = features["tenure"], features["MonthlyCharges"], features["TotalCharges"]
    brand_new_but_billed = (tenure == 0) & (total > 0)
    if brand_new_but_billed.any():
        result.warnings.append(f"{int(brand_new_but_billed.sum())} row(s) have tenure 0 but TotalCharges above 0.")
    expected_total = tenure * monthly
    ratio = total / expected_total.where(expected_total > 0)
    odd_ratio = ratio.notna() & ((ratio < 0.3) | (ratio > 2.0))
    if odd_ratio.any():
        result.warnings.append(
            f"{int(odd_ratio.sum())} row(s) have TotalCharges far from tenure x MonthlyCharges; "
            "please double-check these values."
        )
    no_internet = features["InternetService"] == "No"
    add_on_conflict = no_internet & features[["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                                              "StreamingTV", "StreamingMovies"]].isin(["Yes", "No"]).any(axis=1)
    if add_on_conflict.any():
        result.warnings.append(
            f"{int(add_on_conflict.sum())} row(s) have InternetService = No but list internet add-ons "
            "(expected 'No internet service')."
        )


def validate_customer_frame(df: pd.DataFrame, allow_missing: bool = True,
                            max_rows: int = MAX_BATCH_ROWS) -> ValidationResult:
    """Validate and clean a table of customers.

    Returns model-ready features (only the model columns, standardised values) in
    `result.data` when there are no errors. Missing values are allowed for batch
    files (the pipeline imputes them) but reported as warnings.
    """
    result = ValidationResult()
    if df is None or df.empty:
        result.errors.append("The file contains no rows.")
        return result
    if len(df) > max_rows:
        result.errors.append(f"The file has {len(df):,} rows; the limit is {max_rows:,}.")
        return result

    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        result.errors.append(f"Missing required column(s): {', '.join(missing_columns)}.")
        return result

    cleaned = pd.DataFrame(index=df.index)
    for name in NUMERIC_FEATURES:
        cleaned[name] = _check_numeric(df[name], name, result)
    for name in CATEGORICAL_FEATURES:
        cleaned[name] = _check_categorical(df[name], name, result)

    missing_counts = cleaned.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts) and not allow_missing:
        result.errors.append(f"Missing value(s) in: {', '.join(missing_counts.index)}.")
    elif len(missing_counts):
        details = ", ".join(f"{col} ({n})" for col, n in missing_counts.items())
        result.warnings.append(f"Missing values will be filled using training-set statistics: {details}.")

    if result.errors:
        return result
    _consistency_warnings(cleaned, result)
    result.data = cleaned[FEATURE_COLUMNS]
    return result


def validate_single_customer(record: dict) -> ValidationResult:
    """Validate one customer given as a dict. Every field is required."""
    missing_fields = [col for col in FEATURE_COLUMNS if col not in record]
    if missing_fields:
        return ValidationResult(errors=[f"Missing required field(s): {', '.join(missing_fields)}."])
    return validate_customer_frame(pd.DataFrame([record]), allow_missing=False, max_rows=1)
