"""Claim Pydantic schemas."""

from datetime import date, datetime

from pydantic import BaseModel, Field

from app.models.enums import ClaimStatus, RiskBand


class ClaimBase(BaseModel):
    claim_number: str = Field(..., min_length=1, max_length=100)
    incident_date: date
    reported_date: date | None = None
    claimed_amount: float | None = Field(None, ge=0)
    # Officer decision notes (Phase 9). Nullable on every read shape.
    decision_notes: str | None = None


class ClaimCreate(ClaimBase):
    """Request body for POST /claims."""
    policy_id: int
    vehicle_id: int


class ClaimUpdate(BaseModel):
    status: ClaimStatus | None = None
    risk_band: RiskBand | None = None
    risk_score: float | None = Field(None, ge=0, le=100)
    reported_date: date | None = None
    claimed_amount: float | None = Field(None, ge=0)


class Claim(ClaimBase):
    """Full claim response."""
    id: int
    policy_id: int
    vehicle_id: int
    status: ClaimStatus
    risk_band: RiskBand | None
    risk_score: float | None
    decision_notes: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ClaimSummary(BaseModel):
    """Lightweight claim view for dashboard/list responses (blueprint Section 10)."""
    id: int
    claim_number: str
    status: ClaimStatus
    risk_band: RiskBand | None
    risk_score: float | None
    incident_date: date
    created_at: datetime

    model_config = {"from_attributes": True}


class ClaimDetail(Claim):
    """Extended claim response with nested related data (GET /claims/{id})."""
    # Nested relationships will be populated by Phase 3 API layer
    pass
