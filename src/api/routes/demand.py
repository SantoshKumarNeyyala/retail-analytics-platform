import numpy as np

from fastapi import APIRouter

from src.api.schemas.demand_schema import (
    DemandRequest,
    DemandResponse,
)

from src.api.services.model_loader import demand_model

router = APIRouter(
    prefix="/predict",
    tags=["Demand Forecasting"],
)


@router.post(
    "/demand",
    response_model=DemandResponse,
)
def predict_demand(data: DemandRequest):

    if demand_model is None:

        return {"predicted_sales": 0}

    features = np.array([[data.lag1]])

    prediction = demand_model.predict(features)

    return {"predicted_sales": float(prediction[0])}
