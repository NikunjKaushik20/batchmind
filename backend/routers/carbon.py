from fastapi import APIRouter
from pydantic import BaseModel
from models.carbon import (
    get_per_batch_carbon, get_carbon_summary,
    set_monthly_budget, get_carbon_forecast
)

router = APIRouter(prefix="/api/carbon", tags=["carbon"])


class BudgetRequest(BaseModel):
    budget_kg: float


@router.get("")
def per_batch_carbon():
    return get_per_batch_carbon()


@router.get("/summary")
def carbon_summary():
    return get_carbon_summary()


@router.post("/budget")
def update_budget(req: BudgetRequest):
    return set_monthly_budget(req.budget_kg)


@router.get("/forecast")
def carbon_forecast():
    return get_carbon_forecast()
