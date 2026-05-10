# =========================
# 🤖 ML MODELS + MLFLOW
# =========================

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    classification_report,
)

print("🚀 Starting MLflow Training...")

# -------------------------
# 📥 Load Data
# -------------------------
df = pd.read_parquet("data/silver/retail_features.parquet")
rfm = pd.read_parquet("data/silver/rfm.parquet")

# -------------------------
# 🔬 Set MLflow Experiment
# -------------------------
mlflow.set_experiment("Retail Analytics")

# =========================
# 📊 DEMAND FORECAST MODEL
# =========================
with mlflow.start_run(run_name="Demand Forecast"):

    print("\n📊 Training Demand Model...")

    # Daily sales aggregation
    daily_sales = (
        df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()
    )

    daily_sales.columns = ["Date", "Sales"]

    # Lag feature
    daily_sales["Lag1"] = daily_sales["Sales"].shift(1)

    # Remove null rows
    daily_sales = daily_sales.dropna()

    # Features & target
    X = daily_sales[["Lag1"]]
    y = daily_sales["Sales"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Train model
    model_reg = LinearRegression()
    model_reg.fit(X_train, y_train)

    # Predictions
    y_pred = model_reg.predict(X_test)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log model in MLflow
    mlflow.sklearn.log_model(model_reg, "demand_model")

    # Save model locally for Streamlit
    os.makedirs("src/models", exist_ok=True)

    joblib.dump(model_reg, "src/models/model.pkl")

    print(f"✅ RMSE: {rmse:.2f}")
    print(f"✅ R2 Score: {r2:.2f}")
    print("✅ Demand model saved!")

# =========================
# 👤 CUSTOMER CHURN MODEL
# =========================
with mlflow.start_run(run_name="Churn Model"):

    print("\n👤 Training Churn Model...")

    # Create churn label
    rfm["Churn"] = rfm["Recency"].apply(lambda x: 1 if x > 90 else 0)

    # Features & target
    X = rfm[["Recency", "Frequency", "Monetary"]]
    y = rfm["Churn"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train classifier
    model_clf = LogisticRegression(max_iter=1000)

    model_clf.fit(X_train, y_train)

    # Predictions
    y_pred = model_clf.predict(X_test)

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True)

    accuracy = report["accuracy"]

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model_clf, "churn_model")

    print(f"✅ Accuracy: {accuracy:.2f}")

print("\n🎉 MLflow tracking completed!")
