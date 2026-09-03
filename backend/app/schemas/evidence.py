"""Evidence Pydantic schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import EvidenceType, SignalSeverity


class EvidenceBase(BaseModel):
    evidence_type: EvidenceType
    reference: str | None = Field(None, max_length=500)
    detail_json: dict[str, Any] | None = None


class EvidenceCreate(EvidenceBase):
    risk_signal_id: int


class Evidence(EvidenceBase):
    id: int
    risk_signal_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class RiskSignalWithEvidence(BaseModel):
    """RiskSignal with its linked evidence bundle — used by GET /claims/{id}/evidence."""
    id: int
    rule_id: str
    category: str
    severity: SignalSeverity
    description: str
    created_at: datetime
    evidence: list[Evidence] = Field(default_factory=list)

    model_config = {"from_attributes": True}

