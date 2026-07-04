"""
generate_data.py
-----------------
Generates a realistic synthetic telecom customer churn dataset
(~7,043 rows) modeled on the structure of the classic Telco Customer
Churn dataset referenced in the resume project.

If you have the real Kaggle "Telco Customer Churn" CSV, just drop it
in data/telco_churn.csv and skip running this script.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 7043  # matches "7,000+ row telecom dataset" from resume

genders = np.random.choice(["Male", "Female"], N)
senior_citizen = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])
tenure = np.random.randint(0, 73, N)

phone_service = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines = np.where(
    phone_service == "No",
    "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.42, 0.58]),
)

internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]
)

def dependent_internet_feature(p_yes):
    return np.array([
        "No internet service" if svc == "No"
        else np.random.choice(["Yes", "No"], p=[p_yes, 1 - p_yes])
        for svc in internet_service
    ])

online_security = dependent_internet_feature(0.29)
online_backup = dependent_internet_feature(0.34)
device_protection = dependent_internet_feature(0.34)
tech_support = dependent_internet_feature(0.29)
streaming_tv = dependent_internet_feature(0.38)
streaming_movies = dependent_internet_feature(0.39)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24]
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    N,
    p=[0.34, 0.23, 0.22, 0.21],
)

# Monthly charges depend loosely on services selected (adds realism)
base = 18.0
addon_cost = (
    (internet_service == "DSL") * 15
    + (internet_service == "Fiber optic") * 35
    + (online_security == "Yes") * 5
    + (online_backup == "Yes") * 5
    + (device_protection == "Yes") * 5
    + (tech_support == "Yes") * 5
    + (streaming_tv == "Yes") * 8
    + (streaming_movies == "Yes") * 8
    + (phone_service == "Yes") * 5
)
noise = np.random.normal(0, 5, N)
monthly_charges = np.clip(base + addon_cost + noise, 18, 120).round(2)
total_charges = (monthly_charges * tenure + np.random.normal(0, 20, N)).round(2)
total_charges = np.clip(total_charges, 0, None)

# ---- Churn label with realistic signal baked in ----
churn_score = (
    (contract == "Month-to-month") * 0.35
    + (internet_service == "Fiber optic") * 0.15
    + (tenure < 12) * 0.25
    + (payment_method == "Electronic check") * 0.15
    + (tech_support == "No") * 0.10
    + (online_security == "No") * 0.10
    + (paperless_billing == "Yes") * 0.05
    + (senior_citizen == 1) * 0.05
    - (contract == "Two year") * 0.30
    - (tenure > 48) * 0.20
    + np.random.normal(0, 0.15, N)
)
churn_prob = 1 / (1 + np.exp(-(churn_score - 0.65) * 4))
churn = np.where(np.random.rand(N) < churn_prob, "Yes", "No")

df = pd.DataFrame({
    "customerID": [f"CUST-{i:05d}" for i in range(N)],
    "gender": genders,
    "SeniorCitizen": senior_citizen,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Churn": churn,
})

# Inject a few missing values in TotalCharges (mirrors the real dataset's
# well-known quirk, and justifies the "handling missing values" resume bullet)
missing_idx = np.random.choice(N, 11, replace=False)
df.loc[missing_idx, "TotalCharges"] = np.nan

df.to_csv("data/telco_churn.csv", index=False)
print(f"Generated data/telco_churn.csv with {len(df)} rows")
print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
