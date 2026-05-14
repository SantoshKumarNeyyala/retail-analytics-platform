from fastapi import APIRouter

from src.api.schemas.churn_schema import (
    ChurnRequest,
    ChurnResponse,
)

from src.api.services.model_loader import churn_model

router = APIRouter(
    prefix="/predict",
    tags=["Churn Prediction"],
)


@router.post(
    "/churn",
    response_model=ChurnResponse,
)
def predict_churn(data: ChurnRequest):

    if churn_model is None:

        return {
            "churn_probability": 0,
            "prediction": "Model Not Loaded",
        }

    churn_probability = data.recency / 100

    result = "High Risk" if churn_probability > 0.5 else "Low Risk"

    return {
        "churn_probability": round(churn_probability, 2),
        "prediction": result,
    }
