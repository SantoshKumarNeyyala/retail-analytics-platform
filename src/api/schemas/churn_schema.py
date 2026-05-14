from pydantic import BaseModel


class ChurnRequest(BaseModel):

    recency: float
    frequency: float
    monetary: float


class ChurnResponse(BaseModel):

    churn_probability: float
    prediction: str
