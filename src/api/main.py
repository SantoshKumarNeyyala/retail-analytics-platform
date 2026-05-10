# =========================
# 🚀 FASTAPI ML SERVICE
# =========================

from fastapi import FastAPI
import numpy as np
import mlflow.sklearn

app = FastAPI(title="Retail Analytics API")

print("🚀 Loading models from MLflow...")

# -------------------------
# 🔗 Load models from MLflow
# -------------------------

# You can update run IDs after checking MLflow UI
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

# -------------------------
# 🏠 Home Route
# -------------------------


@app.get("/")
def home():
    return {"message": "Retail Analytics API is running 🚀"}


# -------------------------
# 📈 Demand Prediction
# -------------------------


@app.post("/predict-demand")
def predict_demand(lag1: float):
    if demand_model is None:
        return {"error": "Demand model not loaded"}

    data = np.array([[lag1]])
    prediction = demand_model.predict(data)

    return {"input_lag1": lag1, "predicted_sales": float(prediction[0])}


# -------------------------
# 👤 Churn Prediction
# -------------------------


@app.post("/predict-churn")
def predict_churn(recency: float, frequency: float, monetary: float):
    if churn_model is None:
        return {"error": "Churn model not loaded"}

    data = np.array([[recency, frequency, monetary]])
    prediction = churn_model.predict(data)

    return {
        "recency": recency,
        "frequency": frequency,
        "monetary": monetary,
        "churn": int(prediction[0]),
    }
