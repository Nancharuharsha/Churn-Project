# Customer Churn Prediction System

End-to-end ML pipeline that predicts telecom customer churn, matching the
project described on the resume: data preprocessing → model comparison
(Logistic Regression, Random Forest, XGBoost) → hyperparameter tuning →
deployment as an interactive Streamlit app.

## Project structure

```
churn_project/
├── data/
│   └── telco_churn.csv       # dataset (synthetic, generated below)
├── models/
│   └── churn_model.pkl       # saved trained pipeline (created after training)
├── generate_data.py          # generates a realistic 7,043-row synthetic dataset
├── train_model.py            # preprocessing + model comparison + GridSearchCV tuning
├── app.py                    # Streamlit app for real-time prediction
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Get the data

Option A — use the included generator (synthetic data, same schema as the
real Kaggle "Telco Customer Churn" dataset):

```bash
python generate_data.py
```

Option B — use the real dataset: download "Telco Customer Churn" from
Kaggle and save it as `data/telco_churn.csv` (same column names).

## 2. Train the models

```bash
python train_model.py
```

This will:
- Clean missing values (e.g. blank `TotalCharges`) and encode categorical features
- Train and compare Logistic Regression, Random Forest, and XGBoost
- Run `GridSearchCV` to tune the best-performing model
- Save the final pipeline to `models/churn_model.pkl`

Typical output includes accuracy/ROC-AUC per model and a classification report.

## 3. Run the app

```bash
streamlit run app.py
```

Opens a browser UI where you can fill in a customer's profile (contract
type, tenure, charges, services, etc.) and get an instant churn
prediction with a confidence score.

## Notes

- If `xgboost` isn't installed, `train_model.py` automatically falls back
  to scikit-learn's `GradientBoostingClassifier` so the pipeline still runs.
- The synthetic dataset bakes in realistic churn drivers (month-to-month
  contracts, low tenure, electronic check payment, fiber optic without
  tech support) so model performance and feature importance behave the
  way they would on real telecom data.
- Swap in the real Kaggle dataset any time — `train_model.py` and `app.py`
  don't need any changes since the schema matches.
