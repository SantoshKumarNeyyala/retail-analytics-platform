# ==========================================
# 🚀 WEEKLY MODEL REPORT
# ==========================================

import pandas as pd

print("🚀 Generating Weekly Report...")

report = {
    "Model": [
        "LSTM Forecasting",
        "Prophet Forecasting",
        "Ensemble Forecasting",
        "XGBoost Churn",
        "LightGBM Churn",
        "Customer Segmentation",
        "Price Elasticity",
    ],
    "Metric": ["MAPE", "MAPE", "MAPE", "AUC", "AUC", "Silhouette Score", "R²"],
    "Score": [0.4051, 0.4498, 0.4275, 1.0000, 1.0000, 0.8958, 0.1655],
}

df = pd.DataFrame(report)

print("\n📊 WEEKLY MODEL REPORT")
print(df)

df.to_csv("artifacts/weekly_model_report.csv", index=False)

print("\n✅ Report Saved!")

print("\n🎉 Weekly Report Completed!")
