# Architecture

```
data/raw/telco_customer_churn.csv
        |  src/data_loader.py      load, clean, quality report, (synthetic fallback)
        v
   clean DataFrame
        |  src/preprocessing.py    stratified split, ColumnTransformer
        v
 src/train.py  --> Pipeline(preprocessor + model) x 3, XGBoost tuned
        |            |
        |            +--> src/evaluate.py        metrics, confusion matrix, ROC
        |            +--> src/explainability.py  SHAP values and plots
        v
 models/churn_pipeline.joblib      reports/metrics.json      reports/figures/shap_summary.png
        |                                   |
        v                                   v
 src/predict.py + src/validation.py   app/pages/model_performance.py
        |
        v
 app/ (Streamlit): components/ (theme, ui, charts, cached loaders) + pages/
```

Design rules
- `src/` never imports Streamlit; `app/` never trains a model.
- One place defines columns and allowed values (`src/config.py`), used by training, validation and the forms.
- The preprocessor lives inside the saved pipeline, so inference applies exactly the training transformations.
- Every number on the Model Performance page is read from `reports/metrics.json` written by `src/train.py`.
