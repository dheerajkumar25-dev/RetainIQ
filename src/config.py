"""Central configuration: paths, column definitions, allowed values and constants.

Everything that the training pipeline, the prediction code and the Streamlit
forms must agree on lives here, so the three can never drift apart.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = ROOT_DIR / "docs"

RAW_DATASET_PATH = DATA_RAW_DIR / "telco_customer_churn.csv"
SYNTHETIC_DATASET_PATH = DATA_RAW_DIR / "synthetic_churn.csv"
CLEAN_DATASET_PATH = DATA_PROCESSED_DIR / "churn_clean.csv"

MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
DATASET_INFO_PATH = REPORTS_DIR / "dataset_info.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
SHAP_SUMMARY_IMAGE_PATH = FIGURES_DIR / "shap_summary.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# ---------------------------------------------------------------- experiment settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TUNING_ITERATIONS = 25
CLASSIFICATION_THRESHOLD = 0.5  # probability at/above which we say "likely to churn"

# Risk bands are applied to the predicted probability.
RISK_LOW_UPPER = 0.35
RISK_HIGH_LOWER = 0.65

MAX_BATCH_ROWS = 50_000

# ---------------------------------------------------------------- columns
ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Columns deliberately left out of the model, with the reason.
EXCLUDED_COLUMNS = {
    ID_COLUMN: "Unique identifier; carries no behavioural signal.",
    "gender": "Left out on purpose so the model does not use a demographic attribute "
              "that has no business justification for retention decisions.",
}

YES_NO = ["No", "Yes"]
ADD_ON_OPTIONS = ["No", "Yes", "No internet service"]

CATEGORY_OPTIONS = {
    "SeniorCitizen": YES_NO,
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ADD_ON_OPTIONS,
    "OnlineBackup": ADD_ON_OPTIONS,
    "DeviceProtection": ADD_ON_OPTIONS,
    "TechSupport": ADD_ON_OPTIONS,
    "StreamingTV": ADD_ON_OPTIONS,
    "StreamingMovies": ADD_ON_OPTIONS,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
INTERNET_ADD_ONS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Hard limits: values outside these are rejected as invalid input.
NUMERIC_HARD_LIMITS = {
    "tenure": (0, 120),
    "MonthlyCharges": (0.0, 500.0),
    "TotalCharges": (0.0, 100_000.0),
}
# Range seen in the IBM Telco data: values outside only trigger a warning
# because the model has never seen such customers.
NUMERIC_TYPICAL_RANGE = {
    "tenure": (0, 72),
    "MonthlyCharges": (18.0, 120.0),
    "TotalCharges": (0.0, 8700.0),
}

# ---------------------------------------------------------------- data dictionary
DATA_DICTIONARY = {
    "customerID": ("string", "identifier", "Unique customer identifier."),
    "gender": ("categorical", "excluded", "Customer gender (Female / Male). Not used by the model."),
    "SeniorCitizen": ("categorical", "feature", "Whether the customer is a senior citizen. Stored as 0/1 in the source file, converted to No/Yes."),
    "Partner": ("categorical", "feature", "Whether the customer has a partner."),
    "Dependents": ("categorical", "feature", "Whether the customer has dependents."),
    "tenure": ("numeric", "feature", "Number of months the customer has stayed with the company."),
    "PhoneService": ("categorical", "feature", "Whether the customer has phone service."),
    "MultipleLines": ("categorical", "feature", "Whether the customer has multiple phone lines."),
    "InternetService": ("categorical", "feature", "Type of internet service: DSL, Fiber optic or No."),
    "OnlineSecurity": ("categorical", "feature", "Online security add-on."),
    "OnlineBackup": ("categorical", "feature", "Online backup add-on."),
    "DeviceProtection": ("categorical", "feature", "Device protection add-on."),
    "TechSupport": ("categorical", "feature", "Technical support add-on."),
    "StreamingTV": ("categorical", "feature", "Streaming TV add-on."),
    "StreamingMovies": ("categorical", "feature", "Streaming movies add-on."),
    "Contract": ("categorical", "feature", "Contract term: Month-to-month, One year or Two year."),
    "PaperlessBilling": ("categorical", "feature", "Whether the customer uses paperless billing."),
    "PaymentMethod": ("categorical", "feature", "How the customer pays: electronic check, mailed check, bank transfer or credit card (the last two automatic)."),
    "MonthlyCharges": ("numeric", "feature", "Amount charged to the customer each month."),
    "TotalCharges": ("numeric", "feature", "Total amount charged over the customer's lifetime. Blank for brand-new customers (tenure 0) in the source file; set to 0 during cleaning."),
    "Churn": ("categorical", "target", "Whether the customer left in the last period (Yes / No). This is what the model predicts."),
}
