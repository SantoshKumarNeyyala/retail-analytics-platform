# ==========================================
# 🚀 PRICE ELASTICITY & REVENUE INTELLIGENCE
# ==========================================

import warnings

import os
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ==========================================
# LOAD DATA
# ==========================================

print("🚀 Loading retail dataset...")

df = pd.read_parquet("data/silver/retail_features.parquet")

print(df.head())

# ==========================================
# DATA CLEANING
# ==========================================

df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# ==========================================
# CREATE FEATURES
# ==========================================

print("\n⚙️ Creating elasticity features...")

df["Demand"] = df["Quantity"]

df["Price"] = df["UnitPrice"]

# Promotion Flag
df["Promotion"] = np.where(df["Price"] < df["Price"].median(), 1, 0)

# Revenue
df["Revenue"] = df["Demand"] * df["Price"]

# ==========================================
# LOG TRANSFORMATION
# ==========================================

df["Log_Demand"] = np.log(df["Demand"])

df["Log_Price"] = np.log(df["Price"])

# ==========================================
# PREPARE REGRESSION DATA
# ==========================================

X = df[["Log_Price", "Promotion"]]

X = sm.add_constant(X)

y = df["Log_Demand"]

# ==========================================
# CREATE ARTIFACT FOLDERS
# ==========================================

os.makedirs("artifacts/elasticity", exist_ok=True)

# ==========================================
# SETUP MLFLOW
# ==========================================

mlflow.set_experiment("Retail Analytics")

# ==========================================
# RUN REGRESSION
# ==========================================

with mlflow.start_run(run_name="Price Elasticity Model"):

    print("\n🚀 Training elasticity regression...")

    model = sm.OLS(y, X).fit()

    predictions = model.predict(X)

    r2 = r2_score(y, predictions)

    elasticity = model.params["Log_Price"]

    print("\n📊 MODEL SUMMARY")
    print(model.summary())

    print(f"\n✅ Price Elasticity: {elasticity:.4f}")

    print(f"✅ R² Score: {r2:.4f}")

    # ==========================================
    # LOG METRICS
    # ==========================================

    mlflow.log_metric("price_elasticity", elasticity)

    mlflow.log_metric("r2_score", r2)

    # ==========================================
    # SAVE REGRESSION SUMMARY
    # ==========================================

    summary_path = "artifacts/elasticity/" "regression_summary.txt"

    with open(summary_path, "w") as f:
        f.write(str(model.summary()))

    mlflow.log_artifact(summary_path)

    # ==========================================
    # ELASTICITY VISUALIZATION
    # ==========================================

    plt.figure(figsize=(8, 6))

    plt.scatter(df["Price"], df["Demand"], alpha=0.5)

    plt.xlabel("Price")

    plt.ylabel("Demand")

    plt.title("Price vs Demand")

    elasticity_plot = "artifacts/elasticity/" "price_vs_demand.png"

    plt.savefig(elasticity_plot)

    mlflow.log_artifact(elasticity_plot)

    plt.close()

    # ==========================================
    # WHAT-IF REVENUE SIMULATOR
    # ==========================================

    print("\n💰 Revenue Simulation")

    avg_price = df["Price"].mean()

    avg_demand = df["Demand"].mean()

    scenarios = []

    for pct in [-20, -10, 0, 10, 20]:

        new_price = avg_price * (1 + pct / 100)

        demand_change = elasticity * (pct / 100)

        new_demand = avg_demand * (1 + demand_change)

        projected_revenue = new_price * new_demand

        scenarios.append(
            [
                pct,
                round(new_price, 2),
                round(new_demand, 2),
                round(projected_revenue, 2),
            ]
        )

    scenario_df = pd.DataFrame(
        scenarios,
        columns=[
            "Price Change %",
            "Projected Price",
            "Projected Demand",
            "Projected Revenue",
        ],
    )

    print("\n📊 WHAT-IF SCENARIOS")
    print(scenario_df)

    scenario_path = "artifacts/elasticity/" "revenue_simulation.csv"

    scenario_df.to_csv(scenario_path, index=False)

    mlflow.log_artifact(scenario_path)

print("\n🎉 Price Elasticity Modeling Completed!")
