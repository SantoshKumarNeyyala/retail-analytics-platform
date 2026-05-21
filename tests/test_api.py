import pytest
from httpx import AsyncClient
from httpx import ASGITransport

from src.api.main import app


@pytest.mark.anyio
async def test_home():

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        response = await client.get("/")

    assert response.status_code == 200


@pytest.mark.anyio
async def test_health():

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:

        response = await client.get("/health")

    assert response.status_code == 200
