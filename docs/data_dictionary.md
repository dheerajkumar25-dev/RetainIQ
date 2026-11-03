# Data dictionary

Generated from `src/config.py` by `python -m src.data_loader`.

| Column | Type | Role | Description | Allowed values |
|---|---|---|---|---|
| `customerID` | string | identifier | Unique customer identifier. |  |
| `gender` | categorical | excluded | Customer gender (Female / Male). Not used by the model. |  |
| `SeniorCitizen` | categorical | feature | Whether the customer is a senior citizen. Stored as 0/1 in the source file, converted to No/Yes. | No, Yes |
| `Partner` | categorical | feature | Whether the customer has a partner. | No, Yes |
| `Dependents` | categorical | feature | Whether the customer has dependents. | No, Yes |
| `tenure` | numeric | feature | Number of months the customer has stayed with the company. | any number |
| `PhoneService` | categorical | feature | Whether the customer has phone service. | No, Yes |
| `MultipleLines` | categorical | feature | Whether the customer has multiple phone lines. | No, Yes, No phone service |
| `InternetService` | categorical | feature | Type of internet service: DSL, Fiber optic or No. | DSL, Fiber optic, No |
| `OnlineSecurity` | categorical | feature | Online security add-on. | No, Yes, No internet service |
| `OnlineBackup` | categorical | feature | Online backup add-on. | No, Yes, No internet service |
| `DeviceProtection` | categorical | feature | Device protection add-on. | No, Yes, No internet service |
| `TechSupport` | categorical | feature | Technical support add-on. | No, Yes, No internet service |
| `StreamingTV` | categorical | feature | Streaming TV add-on. | No, Yes, No internet service |
| `StreamingMovies` | categorical | feature | Streaming movies add-on. | No, Yes, No internet service |
| `Contract` | categorical | feature | Contract term: Month-to-month, One year or Two year. | Month-to-month, One year, Two year |
| `PaperlessBilling` | categorical | feature | Whether the customer uses paperless billing. | No, Yes |
| `PaymentMethod` | categorical | feature | How the customer pays: electronic check, mailed check, bank transfer or credit card (the last two automatic). | Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic) |
| `MonthlyCharges` | numeric | feature | Amount charged to the customer each month. | any number |
| `TotalCharges` | numeric | feature | Total amount charged over the customer's lifetime. Blank for brand-new customers (tenure 0) in the source file; set to 0 during cleaning. | any number |
| `Churn` | categorical | target | Whether the customer left in the last period (Yes / No). This is what the model predicts. | Yes, No |
