from fastapi import APIRouter

from src.api.schemas.inventory_schema import (
    InventoryRequest,
    InventoryResponse,
)

router = APIRouter(
    prefix="/inventory",
    tags=["Inventory Optimization"],
)


@router.post(
    "/reorder",
    response_model=InventoryResponse,
)
def inventory_reorder(data: InventoryRequest):

    reorder_point = (data.avg_daily_sales * data.lead_time_days) + data.safety_stock

    if data.current_stock < reorder_point:

        reorder_quantity = int(reorder_point - data.current_stock)

        status = "Reorder Needed"

    else:

        reorder_quantity = 0

        status = "Stock Healthy"

    return {
        "reorder_quantity": reorder_quantity,
        "reorder_status": status,
    }
