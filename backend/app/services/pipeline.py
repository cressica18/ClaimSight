"""Analysis pipeline orchestrator — Phase 11, blueprint Section 12.

Twelve ordered steps, every step isolated so a single failure cannot
stranded the claim in `analyzing`:

  1. Validate claim + required inputs
  2. Set state: claim.status=analyzing, insert Analysis(running)
  3. Run CV on all pending images (per-image isolation)
  4. Extract all pending documents (per-document isolation)
  5. Build ClaimContext (deterministic normalized snapshot)
  6. Run all 9 consistency rules
  7. Compute cost-anomaly features (delegated to risk_engine)
  8. Compute deterministic risk score/band
  9. Generate Evidence for every RiskSignal
 10. Generate/persist Gemini investigation (may return None on failure)
 11. Mark analysis + claim completed
 12. Return PipelineResult

Reliability contract (from user prompt):
- One failed CV image must NOT stop the other images.
- Document extraction failure must NOT stop the whole claim.
- Pure-Python consistency/risk failures MUST mark the claim
  `analysis_failed` rather than leaving it stuck in `analyzing`.
- Gemini failure MUST allow the pipeline to complete with a null
  summary — the claim still gets to `completed` and the
  Investigation row carries `summary_text=None`.
- Database failures must roll back appropriately and restore
  pre-analysis state where possible.
- Concurrent / duplicate analyses are blocked by:
  (a) the in-process lock in `pipeline_locks` (primary, fast path)
  (b) the partial unique index `uq_analyses_one_running_per_claim`
      (multi-process safety net).
- Partial results are NEVER silently discarded. Signals committed
  before a later failure remain in the DB; the Analysis row records
  the failure.
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.analysis import Analysis
from app.models.claim import Claim
from app.models.damage import Damage
from app.models.document import Document
from app.models.evidence import Evidence
from app.models.investigation import Investigation
from app.models.risk_signal import RiskSignal
from app.models.enums import AnalysisStatus, ClaimStatus
from app.services import (
    consistency,
    cv_service,
    document_intelligence,
    evidence as evidence_service,
    gemini_client,
    risk_engine,
)

logger = logging.getLogger(__name__)


# ─── Result type ────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """Snapshot of one run, returned to the API layer.

    `status` is one of AnalysisStatus values. `error_message` is
    populated only on failure. The other fields are populated when
    status == completed and may be partially populated on a failed
    run (e.g. signals that committed before the failure).
    """

    analysis_id: int
    status: str
    claim_id: int
    risk_score: float | None
    risk_band: str | None
    signal_count: int
    evidence_count: int
    investigation_id: int | None
    error_message: str | None = None
    current_step: str | None = None


# ─── Public entry point ─────────────────────────────────────────────────────


def run_analysis(
    claim_id: int,
    db: Session,
    *,
    cv_predictor: Any = None,
    gemini_client_obj: Any = None,
) -> PipelineResult:
    """Run the full pipeline for a claim. Returns a PipelineResult.

    This is the synchronous, single-call entry point used by tests.
    Production uses `init_analysis_row` + `run_analysis_steps` to
    decouple the 202 response from the actual work.

    `db` is the session to use for the *whole* run. The function
    commits and rolls back as needed; it does NOT close the session
    on the way out (caller's responsibility).

    `cv_predictor` and `gemini_client_obj` are injection points used
    by tests; both default to the production service singletons.
    """
    analysis = _init_state(claim_id, db)
    if isinstance(analysis, PipelineResult):
        # Validation failed; _init_state returned a failure result.
        return analysis
    if analysis is None:
        # Another analysis is already running for this claim.
        # _init_state has already raised — this branch is unreachable
        # in practice, kept for type narrowing.
        raise RuntimeError("analysis init returned None")

    return run_analysis_steps(
        claim_id, analysis.id, db,
        cv_predictor=cv_predictor, gemini_client_obj=gemini_client_obj,
    )


def init_analysis_row(claim_id: int, db: Session) -> int:
    """Validate + create the Analysis row. Returns the new analysis_id.

    Used by the API layer to give the client an analysis_id in the
    202 response *before* the actual work begins. Raises on
    validation failure (404 / 409).
    """
    analysis = _init_state(claim_id, db)
    if isinstance(analysis, PipelineResult):
        # Validation failed (e.g. no inputs). The API layer maps
        # this to a 422 — but our pre-check (claim exists, claim
        # not decided) has already passed at this point, so the
        # only remaining cause is missing inputs. The function
        # already wrote a failed Analysis row; the API will
        # return its id.
        return analysis.analysis_id
    return analysis.id


def run_analysis_steps(
    claim_id: int,
    analysis_id: int,
    db: Session,
    *,
    cv_predictor: Any = None,
    gemini_client_obj: Any = None,
) -> PipelineResult:
    """Run steps 3..12 of the pipeline for an analysis row that
    `init_analysis_row` has already created and committed.

    Used by the threaded API worker. Also called by `run_analysis`
    after `init_analysis_row` succeeds.
    """
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise RuntimeError(f"analysis row {analysis_id} disappeared mid-run")
    try:
        return _run_steps(
            claim_id, db, analysis, cv_predictor=cv_predictor,
            gemini_client_obj=gemini_client_obj,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline raised on claim %d", claim_id)
        return _mark_failed(claim_id, db, analysis, exc)


# ─── Step 1+2: validate + create Analysis row ──────────────────────────────


class PipelineValidationError(Exception):
    """Raised by _init_state when the claim is not analysable."""


def _init_state(claim_id: int, db: Session) -> Analysis | PipelineResult:
    """Validate the claim, flip it to `analyzing`, create the
    Analysis row, commit. Returns the Analysis on success, or a
    pre-baked failed PipelineResult if validation fails.

    Raises:
        RuntimeError: if another running analysis already exists
            (the API layer translates this to 409).
    """
    claim = db.get(Claim, claim_id)
    if claim is None:
        raise PipelineValidationError(f"claim {claim_id} not found")
    if claim.status == ClaimStatus.decided.value:
        raise PipelineValidationError("claim is decided; further analysis is not allowed")

    # Step 1: validate required inputs. The pipeline must have at
    # least one image or document to analyze; an empty claim is a
    # user error (Phase 9 lets the user create a claim before
    # uploading anything).
    has_image = db.query(Damage).filter(
        Damage.claim_id == claim_id, Damage.source == "image"
    ).first() is not None
    has_document = db.query(Document).filter(Document.claim_id == claim_id).first() is not None
    if not has_image and not has_document:
        # Create the Analysis row with status=failed so the status
        # endpoint has something to return, then return a failure
        # result. The caller (the API thread) sets the claim status.
        analysis = Analysis(
            claim_id=claim_id,
            status=AnalysisStatus.failed.value,
            current_step=None,
            error_message="no images or documents on claim; nothing to analyze",
        )
        claim.status = ClaimStatus.analysis_failed.value
        db.add(analysis)
        db.add(claim)
        db.commit()
        db.refresh(analysis)
        return PipelineResult(
            analysis_id=analysis.id,
            status=analysis.status,
            claim_id=claim_id,
            risk_score=None,
            risk_band=None,
            signal_count=0,
            evidence_count=0,
            investigation_id=None,
            error_message=analysis.error_message,
            current_step=None,
        )

    # Step 2: set state. The DB-level partial unique index is the
    # authority; if a 'running' row already exists for this claim
    # the INSERT raises IntegrityError. We surface that as a
    # PipelineValidationError-equivalent (HTTP 409) so the API
    # endpoint can return 409.
    analysis = Analysis(
        claim_id=claim_id,
        status=AnalysisStatus.running.value,
        current_step=None,
    )
    claim.status = ClaimStatus.analyzing.value
    db.add(analysis)
    db.add(claim)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        # Find the existing running analysis (if any) for the 409
        # response. The API endpoint looks this up separately, so
        # we just signal a duplicate.
        raise RuntimeError("analysis already running for this claim")
    db.refresh(analysis)
    return analysis


# ─── Steps 3..11: actual work ──────────────────────────────────────────────


def _run_steps(
    claim_id: int,
    db: Session,
    analysis: Analysis,
    *,
    cv_predictor: Any,
    gemini_client_obj: Any,
) -> PipelineResult:
    """Run the per-step pipeline. Any unhandled exception here is
    caught by `run_analysis` and turned into a failure result."""

    # We deliberately do NOT wrap steps 3..10 in a single
    # `with db.begin():` block because each step commits its own
    # work. The blueprint says partial results must be preserved;
    # a single rollback on failure would lose them.

    # Step 3: CV on all pending images.
    # Before re-running CV, clear prior CV-output Damage rows so a
    # rerun does not accumulate duplicate detections. User-uploaded
    # "pending" placeholder rows are preserved (they represent the
    # user's input and are consumed by _run_cv itself).
    _set_step(analysis, "cv", db)
    db.query(Damage).filter(
        Damage.claim_id == claim_id,
        Damage.source == "image",
        Damage.damage_type != "pending",
    ).delete(synchronize_session=False)
    db.commit()
    _run_cv(claim_id, db, predictor=cv_predictor)

    # Step 4: document extraction.
    # Before re-extracting, reset prior Document extraction state so
    # a rerun produces a fresh extraction. The Document row itself
    # (file_path, doc_type) is preserved — only the derived
    # extraction_status, extracted_fields, and raw_confidence are
    # cleared so _run_documents picks it up again.
    _set_step(analysis, "documents", db)
    db.query(Document).filter(
        Document.claim_id == claim_id,
        Document.extraction_status != "pending",
    ).update(
        {
            Document.extraction_status: "pending",
            Document.extracted_fields: None,
            Document.raw_confidence: None,
        },
        synchronize_session=False,
    )
    db.commit()
    _run_documents(claim_id, db)

    # Step 5: build ClaimContext. We also compute the baseline upper
    # bound for the primary damage (matching `_compute_score` in
    # `risk_engine`) so R4 (`r4_excessive_repair_cost`) can fire when
    # the repair-estimate total exceeds it. Without this, R4 would
    # always be silent through the pipeline.
    _set_step(analysis, "context", db)
    # First pass without baseline, so we can pick a primary damage.
    ctx_no_baseline = consistency.build_claim_context(claim_id, db)
    baseline_upper = _compute_baseline_upper(ctx_no_baseline)
    ctx = consistency.build_claim_context(claim_id, db, baseline_upper=baseline_upper)

    # Step 6: consistency rules
    _set_step(analysis, "rules", db)

    # Clean up previous signals for this claim so a rerun doesn't
    # duplicate them. The RiskSignal.evidence relationship declares
    # cascade="all, delete-orphan", but a bulk SQL DELETE bypasses
    # the ORM cascade — Evidence rows would leak on every rerun and
    # the Investigation Summary would re-list the same signal. We
    # therefore delete the linked Evidence rows explicitly before
    # deleting the signals.
    _prior_signal_ids_subq = (
        select(RiskSignal.id)
        .where(RiskSignal.claim_id == claim_id)
        .scalar_subquery()
    )
    db.query(Evidence).filter(
        Evidence.risk_signal_id.in_(_prior_signal_ids_subq)
    ).delete(synchronize_session=False)
    db.query(RiskSignal).filter(
        RiskSignal.claim_id == claim_id
    ).delete(synchronize_session=False)
    db.commit()

    new_signals = consistency.evaluate(ctx)
    persisted_signals = consistency.persist(new_signals, db)

    # Step 7: cost-anomaly features. compute_risk_score internally
    # calls compute_baseline when `baseline=None`, so we do not
    # pre-compute it here. Setting the step marker is enough to
    # make the audit log show this step ran in the right order.
    _set_step(analysis, "risk", db)

    # Step 8: compute deterministic risk score/band
    risk = risk_engine.compute_risk_score(ctx, persisted_signals, baseline=None)

    # Step 8: persist risk score
    claim = db.get(Claim, claim_id)
    risk_engine.persist(risk, db, claim=claim)

    # Step 9: evidence
    _set_step(analysis, "evidence", db)
    evidence_rows = evidence_service.persist_evidence(persisted_signals, ctx, db)

    # Step 10: Gemini. Failure is non-fatal — `persist_investigation`
    # already handles the None case and writes a row with
    # `summary_text=None` so the UI has something to attach to.
    _set_step(analysis, "investigation", db)
    output = gemini_client.generate_investigation(
        claim_id, db, client=gemini_client_obj
    )
    inv = gemini_client.persist_investigation(output, db, claim=claim)

    # Step 11: complete
    claim = db.get(Claim, claim_id)
    claim.status = ClaimStatus.completed.value
    analysis.status = AnalysisStatus.completed.value
    analysis.finished_at = datetime.now(timezone.utc)
    analysis.current_step = None
    db.add(claim)
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return PipelineResult(
        analysis_id=analysis.id,
        status=analysis.status,
        claim_id=claim_id,
        risk_score=float(risk.score) if risk.score is not None else None,
        risk_band=risk.band,
        signal_count=len(persisted_signals),
        evidence_count=len(evidence_rows),
        investigation_id=inv.id if inv else None,
        error_message=None,
        current_step=None,
    )


# ─── Per-step helpers ───────────────────────────────────────────────────────


def _compute_baseline_upper(ctx: consistency.ClaimContext) -> float | None:
    """Compute the baseline upper bound for the primary damage, mirroring
    the logic in `risk_engine._compute_score` (line ~514). Returns None
    when no primary damage is identifiable, in which case R4 stays
    silent (its baseline_missing path).

    Phase 12 fix: without this, the pipeline never populates
    `baseline_upper` on the ClaimContext, so `r4_excessive_repair_cost`
    is always silent and the demo scenarios that depend on R4 cannot
    produce the documented Medium/High bands.
    """
    primary = risk_engine._primary_damage(ctx)
    if primary is None:
        return None
    damage_type, severity = primary
    segment = risk_engine.derive_vehicle_segment(
        ctx.vehicle_make, ctx.vehicle_model, ctx.vehicle_year
    )
    baseline = risk_engine.compute_baseline(segment, damage_type, severity)
    return float(baseline.upper)


def _set_step(analysis: Analysis, step: str, db: Session) -> None:
    """Update the Analysis.current_step field. Commit immediately so
    the GET status endpoint sees the in-flight step even if a later
    step crashes.
    """
    analysis.current_step = step
    db.add(analysis)
    db.commit()


def _run_cv(claim_id: int, db: Session, *, predictor: Any) -> None:
    """Step 3: run CV on every pending image. Per-image isolation.

    `run_cv_on_image` already handles per-image failures by writing
    a Damage row with `damage_type='cv_error'` and returning it. We
    therefore just iterate. If the function itself raises (e.g. the
    model is not loaded at all) we log and continue with the next
    image, marking the failed one with a cv_error row.

    After the run produces at least one new Damage row (success or
    cv_error) for a pending input, we delete the now-stale pending
    row so the per-claim image list reflects a 1:1 mapping between
    uploads and CV results. This is the source of truth the
    frontend uses to render the Claim Analysis stage tracker; if we
    left the pending row in place, a finished claim would always
    show "Image analysis — pending" because there'd be a leftover
    pending row alongside the new analyzed rows.
    """
    pending = db.query(Damage).filter(
        Damage.claim_id == claim_id,
        Damage.source == "image",
        Damage.damage_type == "pending",
    ).all()

    for damage in pending:
        new_damages: list = []
        try:
            new_damages = cv_service.run_cv_on_image(
                db, claim_id, damage.region_ref, predictor=predictor,
            )
            db.commit()
        except Exception as exc:  # noqa: BLE001
            # The cv_service *should* never raise (it writes a
            # cv_error row on failure). If it does, log and
            # synthesize a cv_error row so the claim has a record.
            logger.exception("CV on damage %s raised: %s", damage.id, exc)
            db.rollback()
            _record_cv_error(claim_id, damage.region_ref, db, reason=str(exc)[:500])
            db.commit()

        # If the cv_service produced at least one new row, drop the
        # placeholder pending row so the image list is clean. The
        # cv_service returns [] only in an impossible branch today
        # (every path writes at least one Damage row), but we
        # guard against future changes that would leave the pending
        # row dangling.
        if new_damages:
            try:
                db.delete(damage)
                db.commit()
            except Exception:  # noqa: BLE001
                logger.exception("Failed to delete pending damage %s", damage.id)
                db.rollback()


def _record_cv_error(claim_id: int, region_ref: str | None, db: Session, *, reason: str) -> None:
    """Insert a Damage row marking an image as failed."""
    err = Damage(
        claim_id=claim_id,
        source="image",
        damage_type="cv_error",
        severity="unknown",
        confidence=0.0,
        region_ref=region_ref,
    )
    db.add(err)
    db.flush()


def _run_documents(claim_id: int, db: Session) -> None:
    """Step 4: extract every pending document. Per-document isolation.

    `extract_document` returns True/False and never raises. A False
    return leaves the document in `extraction_status='failed'`, which
    the rules and the frontend both treat as "not extracted" — R9
    will not fire on missing fields.
    """
    pending = db.query(Document).filter(
        Document.claim_id == claim_id,
        Document.extraction_status == "pending",
    ).all()

    for doc in pending:
        try:
            document_intelligence.extract_document(db, claim_id, doc.id)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Doc %s raised during extraction: %s", doc.id, exc)
            db.rollback()
            # Best-effort: flip to failed
            doc.extraction_status = "failed"
            db.add(doc)
            db.commit()


# ─── Failure handling ──────────────────────────────────────────────────────


def _mark_failed(
    claim_id: int,
    db: Session,
    analysis: Analysis,
    exc: BaseException,
) -> PipelineResult:
    """Set claim.status=analysis_failed, analysis.status=failed,
    record the error message. NEVER raises.

    The user-facing invariant is: no claim left stuck in `analyzing`
    after a failure. We therefore swallow any exception this function
    itself might raise (it shouldn't, but tests will be unforgiving).
    """
    try:
        claim = db.get(Claim, claim_id)
        if claim is not None:
            claim.status = ClaimStatus.analysis_failed.value
            db.add(claim)
        analysis.status = AnalysisStatus.failed.value
        analysis.finished_at = datetime.now(timezone.utc)
        # Store the exception type + message; truncate the message so
        # we don't blow past the Text column on a 200KB traceback.
        tb = traceback.format_exception_only(type(exc), exc)
        analysis.error_message = (
            f"{type(exc).__name__}: {exc}"[:2000]
            if not tb
            else f"{type(exc).__name__}: {tb[-1].strip()}"[:2000]
        )
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to mark analysis %s as failed", analysis.id)
        try:
            db.rollback()
        except Exception:  # noqa: BLE001
            pass
    return PipelineResult(
        analysis_id=analysis.id,
        status=AnalysisStatus.failed.value,
        claim_id=claim_id,
        risk_score=None,
        risk_band=None,
        signal_count=0,
        evidence_count=0,
        investigation_id=None,
        error_message=analysis.error_message,
        current_step=analysis.current_step,
    )


# Public exports
__all__ = [
    "PipelineResult",
    "PipelineValidationError",
    "run_analysis",
]
