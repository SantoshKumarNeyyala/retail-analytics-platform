from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from slowapi.middleware import SlowAPIMiddleware
from src.api.security.rate_limit import limiter

from src.api.routes import (
    health,
    churn,
    demand,
    inventory,
)


app = FastAPI(
    title="Enterprise Retail Analytics API",
    version="3.0",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(churn.router)
app.include_router(demand.router)
app.include_router(inventory.router)


@app.get("/")
def root():
    return {"message": "Enterprise Retail Analytics API Running"}


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
