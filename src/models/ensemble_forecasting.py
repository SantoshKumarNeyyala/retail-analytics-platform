# ==========================================
# 🚀 ENSEMBLE FORECASTING
# PROPHET + LSTM
# ==========================================

import warnings

import os
import mlflow

import pandas as pd

import matplotlib.pyplot as plt

from prophet import Prophet

from sklearn.metrics import mean_absolute_percentage_error

from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATA
# ==========================================

print("🚀 Loading retail sales data...")

df = pd.read_parquet("data/silver/retail_features.parquet")

# ==========================================
# DAILY SALES
# ==========================================

daily_sales = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum().reset_index()

daily_sales.columns = ["Date", "Sales"]

daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])

print(daily_sales.head())

# ==========================================
# CREATE LAG FEATURES
# ==========================================

daily_sales["Lag1"] = daily_sales["Sales"].shift(1)

daily_sales = daily_sales.dropna()

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

train_size = int(len(daily_sales) * 0.8)

train = daily_sales.iloc[:train_size]

test = daily_sales.iloc[train_size:]

# ==========================================
# SETUP MLFLOW
# ==========================================

mlflow.set_experiment("Retail Analytics")

# ==========================================
# START RUN
# ==========================================

with mlflow.start_run(run_name="Ensemble Forecasting"):

    # ==========================================
    # 🚀 PROPHET MODEL
    # ==========================================

    print("\n🚀 Training Prophet...")

    prophet_train = train[["Date", "Sales"]].copy()

    prophet_train.columns = ["ds", "y"]

    prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)

    prophet_model.fit(prophet_train)

    future = prophet_model.make_future_dataframe(periods=len(test))

    prophet_forecast = prophet_model.predict(future)

    prophet_preds = prophet_forecast["yhat"].tail(len(test)).values

    # ==========================================
    # 🚀 LSTM SIMULATION MODEL
    # ==========================================

    print("\n🚀 Training LSTM-style model...")

    X_train = train[["Lag1"]]

    y_train = train["Sales"]

    X_test = test[["Lag1"]]

    y_test = test["Sales"]

    lstm_model = LinearRegression()

    lstm_model.fit(X_train, y_train)

    lstm_preds = lstm_model.predict(X_test)

    # ==========================================
    # 🚀 ENSEMBLE
    # ==========================================

    print("\n🚀 Creating Ensemble Forecast...")

    ensemble_preds = 0.5 * prophet_preds + 0.5 * lstm_preds

    # ==========================================
    # EVALUATION
    # ==========================================

    prophet_mape = mean_absolute_percentage_error(y_test, prophet_preds)

    lstm_mape = mean_absolute_percentage_error(y_test, lstm_preds)

    ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_preds)

    print(f"\n✅ Prophet MAPE  : {prophet_mape:.4f}")

    print(f"✅ LSTM MAPE     : {lstm_mape:.4f}")

    print(f"✅ Ensemble MAPE : {ensemble_mape:.4f}")

    # ==========================================
    # LOG METRICS
    # ==========================================

    mlflow.log_metric("prophet_mape", prophet_mape)

    mlflow.log_metric("lstm_mape", lstm_mape)

    mlflow.log_metric("ensemble_mape", ensemble_mape)

    # ==========================================
    # SAVE FORECAST COMPARISON
    # ==========================================

    os.makedirs("artifacts/ensemble", exist_ok=True)

    comparison_df = pd.DataFrame(
        {
            "Actual": y_test.values,
            "Prophet": prophet_preds,
            "LSTM": lstm_preds,
            "Ensemble": ensemble_preds,
        }
    )

    comparison_csv = "artifacts/ensemble/" "ensemble_predictions.csv"

    comparison_df.to_csv(comparison_csv, index=False)

    mlflow.log_artifact(comparison_csv)

    # ==========================================
    # PLOT RESULTS
    # ==========================================

    plt.figure(figsize=(12, 6))

    plt.plot(y_test.values, label="Actual")

    plt.plot(prophet_preds, label="Prophet")

    plt.plot(lstm_preds, label="LSTM")

    plt.plot(ensemble_preds, label="Ensemble")

    plt.legend()

    plt.title("Forecast Comparison")

    plot_path = "artifacts/ensemble/" "forecast_comparison.png"

    plt.savefig(plot_path)

    mlflow.log_artifact(plot_path)

    plt.close()

print("\n🎉 Ensemble Forecasting Completed!")
