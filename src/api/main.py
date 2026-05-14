from fastapi import FastAPI

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

app.include_router(health.router)

app.include_router(churn.router)

app.include_router(demand.router)

app.include_router(inventory.router)


@app.get("/")
def root():

    return {"message": "Enterprise Retail Analytics API Running"}
