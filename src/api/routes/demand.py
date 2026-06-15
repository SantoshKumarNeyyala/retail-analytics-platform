from fastapi import (
    APIRouter,
    Depends,
    Request,
)

import numpy as np

from src.api.schemas.demand_schema import (
    DemandRequest,
    DemandResponse,
)

from src.api.services.model_loader import (
    demand_model,
)

from src.api.security.api_key import (
    validate_api_key,
)

from src.api.security.rate_limit import (
    limiter,
)

from src.api.services.cache_service import (
    get_cached_prediction,
    set_cached_prediction,
)

router = APIRouter(
    prefix="/predict",
    tags=["Demand Forecasting"],
)


@router.post(
    "/demand",
    response_model=DemandResponse,
)
@limiter.limit("100/minute")
def predict_demand(
    request: Request,
    data: DemandRequest,
    api_key: str = Depends(validate_api_key),
):

    cache_key = f"demand_{data.lag1}"

    cached_result = get_cached_prediction(cache_key)

    if cached_result:

        return cached_result

    if demand_model is None:

        return {"predicted_sales": 0}

    features = np.array([[data.lag1]])

    prediction = demand_model.predict(features)

    response = {"predicted_sales": float(prediction[0])}

    set_cached_prediction(
        cache_key,
        response,
    )

    return response
