# RetainIQ: Customer Intelligence & Churn Prediction Platform

**Predict churn. Understand customers. Improve retention.**

RetainIQ is an end-to-end machine learning project that cleans customer data, compares three machine learning models, explains predictions using SHAP, and presents customer insights through a multi-page Streamlit dashboard.

The platform helps identify customers at risk of churn and understand the factors influencing their predicted churn probability.

> **Live demo:** Not deployed yet. The application currently runs locally using Streamlit.

## Screenshots

Screenshots will be added after reviewing all application pages.

| Executive Dashboard | Customer Prediction |
|---|---|
| `docs/images/dashboard.png` | `docs/images/prediction.png` |

| Batch Prediction | Model Performance |
|---|---|
| `docs/images/batch.png` | `docs/images/model.png` |

## Features

- **Executive Dashboard:** KPIs, churn distribution, churn by contract and tenure, monthly charges, customer segments, and data-driven observations.
- **Customer Prediction:** Customer input form with 18 model features, churn probability, Low / Medium / High risk classification, SHAP explanation, and retention considerations.
- **Batch Prediction:** CSV upload, input validation, risk classification, distribution charts, and downloadable prediction results.
- **Customer Analytics:** Interactive filters, churn analysis by contract, internet service and payment method, tenure and charges distributions.
- **Model Performance:** Model comparison, evaluation metrics, confusion matrix, ROC curves, feature importance and SHAP summary.
- **About:** Business problem, project workflow, limitations and data dictionary.

## Technology Stack

- **Programming:** Python 3.12
- **Data Analysis:** Pandas, NumPy, SciPy
- **Machine Learning:** Scikit-learn, XGBoost
- **Explainability:** SHAP
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Deployment/UI:** Streamlit, Docker
- **Testing:** Pytest
- **Model Persistence:** Joblib

No paid services or API keys are required.

## Project Structure

```text
RetainIQ/
├── app/
│   ├── app.py
│   ├── components/
│   │   ├── charts.py
│   │   ├── data_access.py
│   │   ├── theme.py
│   │   └── ui.py
│   ├── pages/
│   │   ├── executive_dashboard.py
│   │   ├── customer_prediction.py
│   │   ├── batch_prediction.py
│   │   ├── customer_analytics.py
│   │   ├── model_performance.py
│   │   └── about.py
│   └── assets/
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   ├── validation.py
│   ├── predict.py
│   └── recommendations.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── churn_pipeline.joblib
├── reports/
│   ├── metrics.json
│   ├── classification_report.txt
│   └── figures/
├── notebooks/
│   └── exploratory_analysis.ipynb
├── tests/
├── docs/
│   ├── architecture.md
│   ├── data_dictionary.md
│   └── images/
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── Dockerfile
├── packages.txt
├── pytest.ini
└── README.md
```

## Dataset

The project uses the **IBM Telco Customer Churn dataset**, containing 7,043 customers and 21 columns.

IBM publishes this dataset as a sample dataset for demonstrating analytics tools. It describes a telecom subscription business. It is a benchmark dataset, not verified customer records from a real company.

- Dataset file: `data/raw/telco_customer_churn.csv`
- Source: `IBM/telco-customer-churn-on-icp4d` GitHub repository.
- Total customers: 7,043
- Churned customers: 1,869
- Retained customers: 5,174
- Churn rate: 26.5%

If the dataset is missing, `python -m src.data_loader` attempts to download it. If downloading fails, the project generates a synthetic dataset and labels it accordingly in the application.

The `customerID` column is dropped, and `gender` is deliberately excluded from the model, leaving 18 input features.

See [docs/data_dictionary.md](docs/data_dictionary.md) for column descriptions.

## Machine Learning Methodology

### 1. Data Cleaning

- Missing `TotalCharges` values for customers with zero tenure are replaced with 0.
- `SeniorCitizen` values are converted from 0/1 to No/Yes.
- No rows are dropped.
- No duplicates were found.

### 2. Train-Test Split

The dataset is divided using a stratified 80/20 split:

- Training samples: 5,634
- Testing samples: 1,409
- Random state: 42

The test set is reserved for final model evaluation.

### 3. Data Preprocessing

A Scikit-learn Pipeline and ColumnTransformer are used.

**Numerical features:**
- Median imputation
- Standard scaling

**Categorical features:**
- Most-frequent imputation
- One-hot encoding
- Unknown categories handled during inference

Preprocessing is fitted on training data only to prevent data leakage.

### 4. Handling Class Imbalance

- Logistic Regression and Random Forest use class weighting.
- XGBoost uses `scale_pos_weight = 2.77`, calculated from the training split.

### 5. Model Training

Three machine learning algorithms are compared:

1. Logistic Regression
2. Random Forest
3. XGBoost

XGBoost is tuned using a 25-iteration randomized search with 3-fold cross-validation and ROC-AUC scoring.

### 6. Model Selection

Models are evaluated using 5-fold cross-validated ROC-AUC on the training data.

XGBoost is selected as the deployed model based on its cross-validation ranking.

### 7. Model Explainability

SHAP TreeExplainer is used to explain individual predictions and identify important features.

SHAP values for one-hot encoded columns are summed by original feature.

### 8. Model Persistence

The complete preprocessing and prediction pipeline is saved using Joblib:

`models/churn_pipeline.joblib`

The Streamlit application loads the saved model rather than retraining it every time.

## Model Evaluation Results

Results generated by `python -m src.train` and stored in `reports/metrics.json`.

The test set contains 1,409 customers and is not used for model fitting or hyperparameter tuning.

**Latest XGBoost results:**

| Metric | Score |
|---|---:|
| CV ROC-AUC | 0.8493 ± 0.0107 |
| Accuracy | 75.73% |
| Precision (Churned) | 52.81% |
| Recall (Churned) | 80.48% |
| F1-score (Churned) | 63.77% |
| Test ROC-AUC | 84.82% |

### Model Comparison

| Model | CV ROC-AUC (Mean ± Std) | Accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost (Deployed)** | **0.8493 ± 0.0107** | **0.757** | **0.528** | **0.805** | **0.638** | **0.848** |
| Logistic Regression | 0.8462 ± 0.0125 | 0.737 | 0.503 | 0.783 | 0.612 | 0.842 |
| Random Forest | 0.8441 ± 0.0091 | 0.767 | 0.545 | 0.741 | 0.628 | 0.840 |

### XGBoost Confusion Matrix

| | Predicted Retained | Predicted Churned |
|---|---:|---:|
| Actual Retained | 766 | 269 |
| Actual Churned | 73 | 301 |

### Understanding the Results

- The three models have relatively close ROC-AUC scores.
- XGBoost has the highest cross-validation ROC-AUC among the three models.
- XGBoost was tuned while the other two models used fixed settings, which slightly favours it.
- Class weighting favours recall over precision. The model identifies most churners but also flags some customers who would have stayed.
- The prediction threshold should be selected based on the business cost of missed churners versus unnecessary retention offers.
- Results may vary slightly when retraining on different machines or library versions.

### Important Features

The top features by mean absolute SHAP value are:

1. Contract
2. Tenure
3. OnlineSecurity
4. InternetService
5. MonthlyCharges
6. PaymentMethod

These features describe what the model uses when making predictions. They should not be interpreted as proof of causation.

## Installation and Setup

### Requirements

- Python 3.12
- Git
- Windows, macOS or Linux

### Windows PowerShell

**1. Clone the repository**

```powershell
git clone https://github.com/dheerajkumar25-dev/RetainIQ.git
cd RetainIQ
```

**2. Create a virtual environment**

```powershell
py -3.12 -m venv .venv
```

**3. Activate the virtual environment**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

**4. Install dependencies**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**5. Prepare the dataset**

```powershell
python -m src.data_loader
```

**6. Train the model (optional)**

The repository already includes a trained model.

```powershell
python -m src.train
```

**7. Launch RetainIQ**

```powershell
python -m streamlit run app/app.py
```

Open the local URL displayed in your terminal, usually:

`http://localhost:8501`

### macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_loader
python -m streamlit run app/app.py
```

## Testing

The project includes a test suite covering:

- Dataset loading and cleaning
- Data validation
- Preprocessing and leakage checks
- Model training and reproducibility
- Model loading
- Single-customer prediction
- Batch prediction
- SHAP explainability
- Missing and invalid inputs
- Streamlit pages and navigation
- CSV upload and validation flows

Run the complete test suite:

```powershell
python -m pytest tests/ -v
```

**Latest local test result: 73 passed.**

## Batch Prediction CSV Format

The batch prediction feature expects the following columns:

```text
tenure
MonthlyCharges
TotalCharges
SeniorCitizen
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

Additional columns such as `customerID` can be retained in the output.

The Batch Prediction page also provides a sample CSV file.

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new application.
4. Select the RetainIQ GitHub repository.
5. Set the main application file to:

   `app/app.py`

6. Select Python 3.12.
7. Deploy the application.

`packages.txt` includes the system dependency required by XGBoost.

### Docker

Build the Docker image:

```bash
docker build -t retainiq .
```

Run the container:

```bash
docker run -p 8501:8501 retainiq
```

Open:

`http://localhost:8501`

## Limitations

- The model uses one sample telecom dataset and may not generalize to other businesses or time periods.
- A random train-test split is used rather than time-based validation.
- Probabilities are not calibrated because class weighting is applied.
- SHAP and feature importance explain model associations, not causation.
- Correlated features may share importance.
- Retention recommendations are rule-based and have not been tested as real interventions.
- No fairness audit has been conducted beyond excluding gender.
- The saved model depends on the pinned Python library versions.

## Future Improvements

- Probability calibration and cost-based threshold selection.
- Time-based model validation.
- Model drift monitoring.
- Fairness analysis across customer groups.
- Uplift modelling to identify effective retention offers.
- REST API using FastAPI.
- Experiment tracking and model versioning.

## Author

**Dheeraj Kumar**

- Institute: IIIT Bhagalpur
- GitHub: [dheerajkumar25-dev](https://github.com/dheerajkumar25-dev)
- Repository: [RetainIQ](https://github.com/dheerajkumar25-dev/RetainIQ)
- Email: dheerajkumar.iiit25@gmail.com

---

*RetainIQ — Turning customer data into actionable retention insights.*