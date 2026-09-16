"""Investigation Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import Recommendation, RiskBand


class InvestigationBase(BaseModel):
    recommendation: Recommendation
    summary_text: str | None = None
    model_version: str | None = Field(None, max_length=100)


class InvestigationCreate(InvestigationBase):
    claim_id: int


class Investigation(InvestigationBase):
    id: int
    claim_id: int
    generated_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class InvestigationSummary(BaseModel):
    """Structured response for GET /claims/{id}/investigation (blueprint Section 7.3)."""
    summary: str | None
    key_concerns: list[str]
    recommendation: Recommendation
    disclaimer: str = "AI-generated, human decision required"
    # The model that produced this summary. Used by the frontend to
    # distinguish demo-mode output (model_version starts with "demo")
    # from real Gemini output, so the UI can label demo summaries
    # honestly instead of presenting them as real model output.
    model_version: str | None = None
    # Authoritative risk result, sourced from the persisted Claim row
    # (Phase 7 risk engine). The frontend uses these for the score/band
    # display on the Investigation Summary page so the user always sees
    # the same value the rest of the app shows — even if the prose
    # summary paraphrases or contradicts it.
    risk_score: float | None = None
    risk_band: RiskBand | None = None

    model_config = {"from_attributes": False}
