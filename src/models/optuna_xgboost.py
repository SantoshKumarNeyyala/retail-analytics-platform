# ==========================================
# 🚀 OPTUNA XGBOOST OPTIMIZATION
# ==========================================

import warnings

import optuna
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATA
# ==========================================

print("🚀 Loading RFM data...")

rfm = pd.read_parquet("data/silver/rfm.parquet")

rfm["Churn"] = rfm["Recency"].apply(lambda x: 1 if x > 90 else 0)

X = rfm[["Recency", "Frequency", "Monetary"]]
y = rfm["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# OPTUNA OBJECTIVE
# ==========================================


def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "eval_metric": "logloss",
        "random_state": 42,
    }

    model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)

    return auc


# ==========================================
# RUN OPTUNA
# ==========================================

print("\n🚀 Running Optuna Optimization...")

study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=20)

print("\n🏆 BEST PARAMETERS")
print(study.best_params)

print(f"\n✅ BEST AUC: {study.best_value:.4f}")

# ==========================================
# TRAIN FINAL MODEL
# ==========================================

best_model = XGBClassifier(**study.best_params)

best_model.fit(X_train, y_train)

# ==========================================
# LOG TO MLFLOW
# ==========================================

mlflow.set_experiment("Retail Analytics")

with mlflow.start_run(run_name="Optuna XGBoost"):

    mlflow.log_params(study.best_params)

    mlflow.log_metric("best_auc", study.best_value)

    mlflow.sklearn.log_model(best_model, "optuna_xgboost_model")

print("\n🎉 Optuna Optimization Completed!")
