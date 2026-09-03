"""Pydantic schemas for the Analysis model — Phase 11."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


AnalysisStatusLiteral = Literal["pending", "running", "completed", "failed"]
ClaimStatusLiteral = Literal["pending", "analyzing", "completed", "analysis_failed", "decided"]


class AnalysisStartResponse(BaseModel):
    """Response body for POST /claims/{id}/analyze (HTTP 202)."""
    analysis_id: int
    status: AnalysisStatusLiteral
    claim_id: int


class AnalysisResultSummary(BaseModel):
    """The `result` block in the status response. Populated only when
    the analysis has reached a terminal state. For failed analyses
    the caller should look at `error_message` instead.
    """
    risk_score: float | None
    risk_band: str | None
    signal_count: int
    evidence_count: int
    investigation_id: int | None


class AnalysisStatusResponse(BaseModel):
    """Response body for GET /claims/{id}/analysis/{analysis_id}."""
    analysis_id: int
    claim_id: int
    status: AnalysisStatusLiteral
    current_step: str | None
    started_at: datetime
    finished_at: datetime | None
    error_message: str | None
    claim_status: ClaimStatusLiteral
    result: AnalysisResultSummary | None = None
