# ==========================================
# 🚀 STACKING CHURN MODEL
# XGBOOST + LIGHTGBM + RANDOM FOREST
# META LEARNER = LOGISTIC REGRESSION
# ==========================================

import warnings


import os
import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATA
# ==========================================

print("🚀 Loading RFM data...")

rfm = pd.read_parquet("data/silver/rfm.parquet")

print(rfm.head())

# ==========================================
# CREATE CHURN LABEL
# ==========================================

rfm["Churn"] = rfm["Recency"].apply(lambda x: 1 if x > 90 else 0)

# ==========================================
# FEATURES
# ==========================================

X = rfm[["Recency", "Frequency", "Monetary"]]

y = rfm["Churn"]

# ==========================================
# SCALE FEATURES
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# SMOTE BALANCING
# ==========================================

print("\n⚖️ Applying SMOTE balancing...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ==========================================
# BASE MODELS
# ==========================================

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

lgbm_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

rf_model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)

# ==========================================
# STACKING MODEL
# ==========================================

print("\n🚀 Building stacking ensemble...")

stack_model = StackingClassifier(
    estimators=[("xgb", xgb_model), ("lgbm", lgbm_model), ("rf", rf_model)],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1,
)

# ==========================================
# MLFLOW
# ==========================================

mlflow.set_experiment("Retail Analytics")

with mlflow.start_run(run_name="Stacking Churn Ensemble"):

    print("\n🚀 Training stacking model...")

    stack_model.fit(X_train_smote, y_train_smote)

    # Predictions
    y_pred = stack_model.predict(X_test)

    y_prob = stack_model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred)

    print("\n📊 STACKING MODEL RESULTS")
    print(report)

    print(f"✅ Ensemble AUC: {auc:.4f}")

    # MLflow logging
    mlflow.log_metric("auc", auc)

    mlflow.sklearn.log_model(stack_model, "stacking_churn_model")

# ==========================================
# SAVE PREDICTIONS
# ==========================================

os.makedirs("artifacts/predictions", exist_ok=True)

predictions = pd.DataFrame(
    {"Actual": y_test, "Predicted": y_pred, "Probability": y_prob}
)

predictions.to_csv("artifacts/predictions/stacking_predictions.csv", index=False)

print("\n💾 Predictions saved!")

print("\n🎉 Stacking Churn Modeling Completed!")
