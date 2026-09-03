"""RiskSignal Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import SignalSeverity


class RiskSignalBase(BaseModel):
    rule_id: str = Field(..., min_length=1, max_length=100)
    category: str = Field(..., min_length=1, max_length=100)
    severity: SignalSeverity
    description: str


class RiskSignalCreate(RiskSignalBase):
    claim_id: int


class RiskSignal(RiskSignalBase):
    id: int
    claim_id: int
    created_at: datetime

    model_config = {"from_attributes": True}
