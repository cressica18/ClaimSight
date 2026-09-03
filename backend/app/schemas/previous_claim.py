"""PreviousClaim Pydantic schemas."""

from datetime import date, datetime

from pydantic import BaseModel, Field


class PreviousClaimBase(BaseModel):
    claim_number: str = Field(..., min_length=1, max_length=100)
    incident_date: date
    damage_summary: str | None = None
    claimed_amount: float | None = Field(None, ge=0)
    overlap_score: float | None = Field(None, ge=0.0, le=1.0)


class PreviousClaimCreate(PreviousClaimBase):
    customer_id: int
    vehicle_id: int


class PreviousClaim(PreviousClaimBase):
    id: int
    customer_id: int
    vehicle_id: int
    created_at: datetime

    model_config = {"from_attributes": True}
