# ==========================================
# 🚀 ADVANCED CHURN PREDICTION
# XGBOOST + LIGHTGBM + SHAP
# ==========================================

import warnings

import os
import mlflow
import mlflow.sklearn

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.over_sampling import SMOTE

import shap

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
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# HANDLE IMBALANCE
# ==========================================

print("⚖️ Applying SMOTE balancing...")

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# ==========================================
# SETUP MLFLOW
# ==========================================

mlflow.set_experiment("Retail Analytics")

# ==========================================
# XGBOOST MODEL
# ==========================================

with mlflow.start_run(run_name="XGBoost Churn Model"):

    print("\n🚀 Training XGBoost...")

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        base_score=0.5,
        random_state=42,
    )

    xgb_model.fit(X_train_smote, y_train_smote)

    # Predictions
    y_pred = xgb_model.predict(X_test)

    y_prob = xgb_model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred)

    print("\n📊 XGBoost Results")
    print(report)

    print(f"✅ AUC Score: {auc:.4f}")

    # MLflow Logging
    mlflow.log_metric("auc", auc)

    mlflow.sklearn.log_model(xgb_model, "xgboost_churn_model")

    # ==========================================
    # SAVE CONFUSION MATRIX
    # ==========================================

    os.makedirs("artifacts/plots", exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 5))

    plt.imshow(cm)

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.colorbar()

    cm_path = "artifacts/plots/confusion_matrix.png"

    plt.savefig(cm_path)

    mlflow.log_artifact(cm_path)

    # ==========================================
    # SHAP EXPLAINABILITY
    # ==========================================

    print("\n🧠 Generating SHAP Explainability...")

    os.makedirs("artifacts/shap", exist_ok=True)

    explainer = shap.TreeExplainer(xgb_model.get_booster())

    shap_values = explainer.shap_values(X_test)

    # SHAP Summary Plot
    shap.summary_plot(shap_values, X_test, show=False)

    shap_summary_path = "artifacts/shap/shap_summary.png"

    plt.savefig(shap_summary_path, bbox_inches="tight")

    mlflow.log_artifact(shap_summary_path)

    plt.close()

    # SHAP Bar Plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)

    shap_bar_path = "artifacts/shap/shap_bar.png"

    plt.savefig(shap_bar_path, bbox_inches="tight")

    mlflow.log_artifact(shap_bar_path)

    plt.close()

# ==========================================
# LIGHTGBM MODEL
# ==========================================

with mlflow.start_run(run_name="LightGBM Churn Model"):

    print("\n🚀 Training LightGBM...")

    lgbm_model = LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=42)

    lgbm_model.fit(X_train_smote, y_train_smote)

    y_pred_lgbm = lgbm_model.predict(X_test)

    y_prob_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

    auc_lgbm = roc_auc_score(y_test, y_prob_lgbm)

    print(f"✅ LightGBM AUC: {auc_lgbm:.4f}")

    mlflow.log_metric("auc", auc_lgbm)

    mlflow.sklearn.log_model(lgbm_model, "lightgbm_churn_model")

# ==========================================
# MODEL COMPARISON
# ==========================================

print("\n📊 MODEL COMPARISON")

print(f"XGBoost AUC  : {auc:.4f}")
print(f"LightGBM AUC : {auc_lgbm:.4f}")

best_model = "XGBoost" if auc > auc_lgbm else "LightGBM"

print(f"\n🏆 Best Model: {best_model}")

print("\n🎉 Churn Modeling Completed!")
