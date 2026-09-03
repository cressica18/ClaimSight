"""Repair Estimate and Repair Item Pydantic schemas."""

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.models.enums import RepairOperation


class RepairItemBase(BaseModel):
    part_name: str | None = Field(None, max_length=255)
    operation: RepairOperation | None = None
    cost: float | None = Field(None, ge=0)
    labor_hours: float | None = Field(None, ge=0)


class RepairItemCreate(RepairItemBase):
    pass


class RepairItem(RepairItemBase):
    id: int
    repair_estimate_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class RepairEstimateBase(BaseModel):
    shop_name: str | None = Field(None, max_length=255)
    total_cost: float | None = Field(None, ge=0)
    currency: str = Field("USD", max_length=3)
    issued_date: date | None = None


class RepairEstimateCreate(RepairEstimateBase):
    claim_id: int
    document_id: int | None = None
    items: list[RepairItemCreate] = Field(default_factory=list)


class RepairEstimate(RepairEstimateBase):
    id: int
    claim_id: int
    document_id: int | None
    items: list[RepairItem] = Field(default_factory=list)
    created_at: datetime

    model_config = {"from_attributes": True}
