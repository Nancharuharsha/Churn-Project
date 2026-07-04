"""
train_model.py
---------------
End-to-end ML pipeline for the Customer Churn Prediction project.

- Loads data/telco_churn.csv
- Cleans missing values, encodes categoricals, scales numeric features
- Trains & compares Logistic Regression, Random Forest, and XGBoost
- Tunes the best-performing model with GridSearchCV
- Saves the final pipeline (preprocessing + model) to models/churn_model.pkl
  so the Streamlit app can load it directly.
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[warning] xgboost not installed — will fall back to "
          "GradientBoostingClassifier. Run `pip install xgboost` for the "
          "full comparison described in the resume.")
    from sklearn.ensemble import GradientBoostingClassifier

RANDOM_STATE = 42


def load_data(path="data/telco_churn.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["customerID"])
    # TotalCharges sometimes arrives as blank strings in the real dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return preprocessor


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"\n{name}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))
    return acc, auc


def main():
    df = load_data()
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    }
    if HAS_XGB:
        candidates["XGBoost"] = XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE
        )
    else:
        candidates["Gradient Boosting"] = GradientBoostingClassifier(random_state=RANDOM_STATE)

    results = {}
    fitted = {}
    for name, clf in candidates.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        acc, auc = evaluate(name, pipe, X_test, y_test)
        results[name] = acc
        fitted[name] = pipe

    best_name = max(results, key=results.get)
    print(f"\nBest baseline model: {best_name} (accuracy={results[best_name]:.4f})")

    # ---- Hyperparameter tuning on the best model ----
    print(f"\nRunning GridSearchCV on {best_name}...")
    best_pipe = fitted[best_name]

    if best_name == "Logistic Regression":
        param_grid = {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "liblinear"],
        }
    elif best_name == "Random Forest":
        param_grid = {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
        }
    elif best_name == "XGBoost":
        param_grid = {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1],
        }
    else:  # Gradient Boosting fallback
        param_grid = {
            "classifier__n_estimators": [200, 400],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.01, 0.1],
        }

    grid = GridSearchCV(best_pipe, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params: {grid.best_params_}")
    tuned_acc, tuned_auc = evaluate(f"{best_name} (tuned)", grid.best_estimator_, X_test, y_test)

    joblib.dump(grid.best_estimator_, "models/churn_model.pkl")
    print(f"\nSaved final tuned model to models/churn_model.pkl "
          f"(test accuracy={tuned_acc:.4f})")


if __name__ == "__main__":
    main()
