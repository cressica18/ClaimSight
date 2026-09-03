"""Analysis pipeline API — Phase 11, blueprint Sections 12 + 13.

Three endpoints:
  POST /claims/{id}/analyze          → 202 + {analysis_id, status, claim_id}
  GET  /claims/{id}/analysis/{aid}   → 200 + AnalysisStatusResponse
  GET  /claims/{id}/analysis/latest  → 200 + AnalysisStatusResponse (404 if none)

The POST handler returns 202 immediately and runs the pipeline in a
background thread. The thread owns its own DB session so the HTTP
session can return to the client without holding a connection.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from app.api import deps
from app.db.session import SessionLocal
from app.models.analysis import Analysis
from app.models.claim import Claim
from app.models.enums import AnalysisStatus, ClaimStatus
from app.models.evidence import Evidence
from app.models.investigation import Investigation
from app.models.risk_signal import RiskSignal
from app.schemas.analysis import (
    AnalysisResultSummary,
    AnalysisStartResponse,
    AnalysisStatusResponse,
)
from app.services import pipeline as pipeline_service
from app.services import pipeline_locks

logger = logging.getLogger(__name__)

router = APIRouter()


# ─── POST /claims/{id}/analyze ──────────────────────────────────────────────


@router.post(
    "/{id}/analyze",
    response_model=AnalysisStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start a full analysis run on a claim",
)
def start_analysis(
    id: int,
    db: Session = Depends(deps.get_db),
) -> Any:
    """Kick off the pipeline.

    Returns 202 + analysis_id once the Analysis row is created and
    the claim is flipped to status='analyzing'. The actual work
    continues in a background thread; clients poll
    GET /claims/{id}/analysis/{analysis_id} until status is
    'completed' or 'failed'.

    Error responses:
      404 — claim not found
      409 — claim is decided, OR an analysis is already running
    """
    claim = db.get(Claim, id)
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found.",
        )
    if claim.status == ClaimStatus.decided.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Claim has been decided; further analysis is not allowed.",
        )

    # Acquire the in-process lock BEFORE creating the Analysis row.
    # This serializes overlapping POSTs within one process; the DB
    # partial unique index is the multi-process safety net.
    if not pipeline_locks.acquire(id):
        existing = _latest_running_analysis(id, db)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Analysis already in progress.",
                "analysis_id": existing.id if existing else None,
            },
        )

    # Create the Analysis row synchronously so the request can
    # return 202 + analysis_id immediately. The actual pipeline work
    # (steps 3..12) runs in a background thread.
    try:
        analysis_id = pipeline_service.init_analysis_row(id, db)
    except Exception:  # noqa: BLE001
        pipeline_locks.release(id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start analysis.",
        )

    # Spawn a background thread to actually run the pipeline. The
    # thread gets its own session; FastAPI's `db` is closed by the
    # time the response is sent, and we do not want the worker
    # to share the request-scoped session.
    thread = threading.Thread(
        target=_run_in_thread,
        args=(id, analysis_id),
        name=f"analysis-{id}",
        daemon=True,
    )
    thread.start()

    return AnalysisStartResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.running.value,
        claim_id=id,
    )


def _run_in_thread(claim_id: int, analysis_id: int) -> None:
    """Background worker: runs the pipeline with a fresh session and
    releases the lock when done, no matter the outcome.
    """
    db = SessionLocal()
    try:
        try:
            pipeline_service.run_analysis_steps(claim_id, analysis_id, db)
        except Exception:  # noqa: BLE001
            logger.exception("Pipeline thread crashed for claim %d", claim_id)
    finally:
        try:
            db.close()
        except Exception:  # noqa: BLE001
            pass
        pipeline_locks.release(claim_id)


# ─── GET /claims/{id}/analysis/latest ───────────────────────────────────────
# MUST be declared before `/{id}/analysis/{aid}` so that FastAPI's
# route matcher picks the literal `latest` path over the typed
# `{aid: int}` parameter. Otherwise /analysis/latest tries to parse
# the string "latest" as an int and 422s before the handler runs.


@router.get(
    "/{id}/analysis/latest",
    response_model=AnalysisStatusResponse,
    summary="Get the most recent analysis for a claim",
)
def get_latest_analysis(
    id: int,
    db: Session = Depends(deps.get_db),
) -> Any:
    claim = db.get(Claim, id)
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found.",
        )
    stmt = (
        select(Analysis)
        .where(Analysis.claim_id == id)
        .order_by(desc(Analysis.started_at))
        .limit(1)
    )
    analysis = db.execute(stmt).scalars().first()
    if analysis is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No analysis has been run on this claim.",
        )
    return _to_status_response(analysis, claim, db)


# ─── GET /claims/{id}/analysis/{aid} ────────────────────────────────────────


@router.get(
    "/{id}/analysis/{aid}",
    response_model=AnalysisStatusResponse,
    summary="Get status of one analysis run",
)
def get_analysis_status(
    id: int,
    aid: int,
    db: Session = Depends(deps.get_db),
) -> Any:
    """Return the current status of analysis `aid` for claim `id`.

    404 if either the claim or the analysis is missing, or if the
    analysis does not belong to the claim.
    """
    claim = db.get(Claim, id)
    if claim is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found.",
        )
    analysis = db.get(Analysis, aid)
    if analysis is None or analysis.claim_id != id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found for this claim.",
        )
    return _to_status_response(analysis, claim, db)


# ─── Helpers ────────────────────────────────────────────────────────────────


def _latest_running_analysis(claim_id: int, db: Session) -> Analysis | None:
    stmt = (
        select(Analysis)
        .where(
            Analysis.claim_id == claim_id,
            Analysis.status == AnalysisStatus.running.value,
        )
        .order_by(desc(Analysis.started_at))
        .limit(1)
    )
    return db.execute(stmt).scalars().first()


def _to_status_response(
    analysis: Analysis, claim: Claim, db: Session
) -> AnalysisStatusResponse:
    """Build the full status response, including the result block
    if the analysis has completed.
    """
    result = None
    if analysis.status == AnalysisStatus.completed.value:
        # Compute counts on the fly — they are cheap and a
        # denormalized counter would go stale.
        signal_count = db.query(RiskSignal).filter(
            RiskSignal.claim_id == claim.id
        ).count()
        evidence_count = (
            db.query(Evidence)
            .join(RiskSignal, Evidence.risk_signal_id == RiskSignal.id)
            .filter(RiskSignal.claim_id == claim.id)
            .count()
        )
        inv = db.query(Investigation).filter(Investigation.claim_id == claim.id).first()
        result = AnalysisResultSummary(
            risk_score=float(claim.risk_score) if claim.risk_score is not None else None,
            risk_band=claim.risk_band,
            signal_count=signal_count,
            evidence_count=evidence_count,
            investigation_id=inv.id if inv else None,
        )
    return AnalysisStatusResponse(
        analysis_id=analysis.id,
        claim_id=claim.id,
        status=analysis.status,
        current_step=analysis.current_step,
        started_at=analysis.started_at,
        finished_at=analysis.finished_at,
        error_message=analysis.error_message,
        claim_status=claim.status,
        result=result,
    )
