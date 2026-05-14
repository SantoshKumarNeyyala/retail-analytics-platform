import mlflow.sklearn

print("🚀 Loading models from MLflow...")

DEMAND_MODEL_URI = "runs:/b60c248074c54e9398e75e906b3ced69/demand_model"

CHURN_MODEL_URI = "runs:/a811e18505904be3a036fe6ec672215c/churn_model"

try:

    demand_model = mlflow.sklearn.load_model(DEMAND_MODEL_URI)

    churn_model = mlflow.sklearn.load_model(CHURN_MODEL_URI)

    print("✅ Models loaded successfully")

except Exception as e:

    print("⚠️ Error loading models:", e)

    demand_model = None
    churn_model = None
