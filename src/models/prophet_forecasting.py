# ==========================================
# 🚀 PROPHET FORECASTING MODEL
# ==========================================

import warnings


import os
import mlflow

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from prophet import Prophet

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

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

daily_sales.columns = ["ds", "y"]

daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])

print(daily_sales.head())

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
# PROPHET MODEL
# ==========================================

with mlflow.start_run(run_name="Prophet Forecasting"):

    print("\n🚀 Training Prophet model...")

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False
    )

    model.fit(train)

    # ==========================================
    # FORECAST
    # ==========================================

    future = model.make_future_dataframe(periods=len(test))

    forecast = model.predict(future)

    predictions = forecast[["ds", "yhat"]].tail(len(test))

    # ==========================================
    # EVALUATION
    # ==========================================

    y_true = test["y"].values

    y_pred = predictions["yhat"].values

    mape = mean_absolute_percentage_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n✅ Prophet MAPE: {mape:.4f}")

    print(f"✅ Prophet RMSE: {rmse:.2f}")

    # ==========================================
    # LOG METRICS
    # ==========================================

    mlflow.log_metric("mape", mape)

    mlflow.log_metric("rmse", rmse)

    # ==========================================
    # SAVE FORECAST PLOT
    # ==========================================

    os.makedirs("artifacts/forecasting", exist_ok=True)

    fig1 = model.plot(forecast)

    forecast_plot_path = "artifacts/forecasting/" "prophet_forecast.png"

    plt.savefig(forecast_plot_path)

    mlflow.log_artifact(forecast_plot_path)

    plt.close()

    # ==========================================
    # COMPONENTS PLOT
    # ==========================================

    fig2 = model.plot_components(forecast)

    components_plot_path = "artifacts/forecasting/" "prophet_components.png"

    plt.savefig(components_plot_path)

    mlflow.log_artifact(components_plot_path)

    plt.close()

    # ==========================================
    # SAVE FORECAST CSV
    # ==========================================

    forecast_output = predictions.copy()

    forecast_output["Actual"] = y_true

    forecast_csv_path = "artifacts/forecasting/" "prophet_predictions.csv"

    forecast_output.to_csv(forecast_csv_path, index=False)

    mlflow.log_artifact(forecast_csv_path)

print("\n🎉 Prophet Forecasting Completed!")
