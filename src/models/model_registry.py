# ==========================================
# 🚀 MODEL REGISTRY
# ==========================================

import mlflow
from mlflow.tracking import MlflowClient

print("🚀 Connecting to MLflow...")

client = MlflowClient()

experiment = mlflow.get_experiment_by_name("Retail Analytics")

runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

print("\n📊 AVAILABLE RUNS")
print(runs[["run_id", "tags.mlflow.runName"]])

# ==========================================
# REGISTER BEST MODEL
# ==========================================

best_run_id = runs.iloc[0]["run_id"]

model_uri = f"runs:/{best_run_id}/xgboost_churn_model"

print("\n🚀 Registering Model...")

latest_run_id = "5e4c303883824a9ebee9b9a5cdee8cb0"

model_uri = f"runs:/{latest_run_id}/xgboost_churn_model"

registered_model = mlflow.register_model(model_uri=model_uri, name="Retail_Churn_Model")

print("\n✅ Model Registered Successfully!")

print(f"Model Name: {registered_model.name}")
print(f"Version: {registered_model.version}")

print("\n🎉 Model Registry Completed!")
