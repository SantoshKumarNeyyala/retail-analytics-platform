from fastapi import (
    APIRouter,
    Depends,
    Request,
)

from src.api.schemas.churn_schema import (
    ChurnRequest,
    ChurnResponse,
)

from src.api.services.model_loader import (
    churn_model,
)

from src.api.security.api_key import (
    validate_api_key,
)

from src.api.security.rate_limit import (
    limiter,
)

router = APIRouter(
    prefix="/predict",
    tags=["Churn Prediction"],
)


@router.post(
    "/churn",
    response_model=ChurnResponse,
)
@limiter.limit("100/minute")
def predict_churn(
    request: Request,
    data: ChurnRequest,
    api_key: str = Depends(validate_api_key),
):

    if churn_model is None:

        return {
            "churn_probability": 0,
            "prediction": "Model Not Loaded",
        }

    churn_probability = data.recency / 100

    result = "High Risk" if churn_probability > 0.5 else "Low Risk"

    return {
        "churn_probability": round(
            churn_probability,
            2,
        ),
        "prediction": result,
    }
