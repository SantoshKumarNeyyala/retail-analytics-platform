from pydantic import BaseModel


class InventoryRequest(BaseModel):

    current_stock: int
    avg_daily_sales: float
    lead_time_days: int
    safety_stock: int


class InventoryResponse(BaseModel):

    reorder_quantity: int
    reorder_status: str
