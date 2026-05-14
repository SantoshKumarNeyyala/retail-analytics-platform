from pydantic import BaseModel


class DemandRequest(BaseModel):

    lag1: float


class DemandResponse(BaseModel):

    predicted_sales: float
