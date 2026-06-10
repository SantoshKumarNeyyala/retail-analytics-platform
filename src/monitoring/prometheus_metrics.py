from prometheus_client import Gauge

model_drift_score = Gauge(
    "model_drift_score",
    "Current model drift score",
)

model_mape = Gauge(
    "model_mape",
    "Current model MAPE",
)

model_drift_score.set(0.12)
model_mape.set(8.4)
