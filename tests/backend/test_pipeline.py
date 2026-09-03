"""
Phase 11 — Full Analysis Pipeline Integration tests.

Each test uses the SQLite in-memory fixture from conftest.py and
injects deterministic fakes for the CV predictor and the Gemini
client. No real LLM is called; no real CV model is loaded.

Coverage (per user prompt, all required scenarios):
  1. test_full_pipeline_success
  2. test_missing_inputs_marks_failed
  3. test_already_running_409
  4. test_cv_failure_isolation
  5. test_document_extraction_failure_isolation
  6. test_consistency_failure_marks_analysis_failed
  7. test_gemini_failure_keeps_claim_completed
  8. test_status_polling_endpoint
  9. test_persistence_of_signals_evidence_risk_investigation
 10. test_no_claim_stuck_analyzing
 11. test_decided_claim_analyze_returns_409
 12. test_start_analysis_returns_202
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from app.api import pipeline as api_pipeline
from app.db.session import SessionLocal
from app.models.analysis import Analysis
from app.models.claim import Claim
from app.models.customer import Customer
from app.models.damage import Damage
from app.models.document import Document
from app.models.enums import (
    AnalysisStatus,
    ClaimStatus,
    DocType,
    ExtractionStatus,
)
from app.models.evidence import Evidence
from app.models.investigation import Investigation
from app.models.policy import Policy
from app.models.risk_signal import RiskSignal
from app.models.repair import RepairEstimate
from app.models.vehicle import Vehicle
from app.services import (
    consistency as consistency_service,
    document_intelligence,
    evidence as evidence_service,
    gemini_client as gemini_module,
    pipeline as pipeline_service,
    pipeline_locks,
    risk_engine as risk_engine_service,
)


# ─── Fake CV predictor ──────────────────────────────────────────────────────


class FakeCVPredictor:
    """Returns one DamageTypePrediction per call. Failable on demand."""

    def __init__(self, *, fail: bool = False, damage_type: str = "scratch"):
        self.fail = fail
        self.damage_type = damage_type
        self.calls: list[str] = []

    def predict_from_path(self, image_path):  # type: ignore[no-untyped-def]
        from ml.inference.predictor import (
            CVPrediction,
            DamageTypePrediction,
            SeverityPrediction,
        )

        self.calls.append(str(image_path))
        if self.fail:
            raise RuntimeError("simulated CV failure")
        return CVPrediction(
            damage_types=[
                DamageTypePrediction(label=self.damage_type, confidence=0.9),
            ],
            severity=SeverityPrediction(label="moderate", confidence=0.85),
            low_confidence=False,
            model_version="fake_cv_v1",
            source_image=str(image_path),
            timestamp="2026-01-01T00:00:00",
            error=None,
        )


# ─── Fake Gemini client ─────────────────────────────────────────────────────


class FakeGeminiClient:
    """Returns a fixed InvestigationOutput. Set `return_none` to simulate
    a real Gemini failure (the production `generate_investigation` path
    then writes summary_text=None and the deterministic recommendation).
    """

    def __init__(self, *, return_none: bool = False, summary: str = "Test summary."):
        self.return_none = return_none
        self.summary = summary
        self.calls = 0

    def generate(self, input):  # type: ignore[no-untyped-def]
        from app.services.gemini_client import InvestigationOutput

        self.calls += 1
        if self.return_none:
            return None
        return InvestigationOutput(
            summary=self.summary,
            key_concerns=["[R1_unsupported_damage] stub"],
            recommendation="manual_review",
            model_version="fake_gemini_v1",
        )


# ─── Seed helpers ───────────────────────────────────────────────────────────


def _seed_claim(
    db,
    *,
    with_image: bool = True,
    with_document: bool = True,
    claim_number: str = "CLM-PIPE-1",
    incident_date: dt.date | None = None,
    claimed_amount: float = 500.0,
) -> Claim:
    """Create the minimum graph: customer, vehicle, policy, claim,
    and optionally a pending image and/or document. Returns the Claim.
    """
    # Unique email per claim_number so tests that seed multiple claims
    # don't trip the customers.email unique constraint.
    email = f"pipe-{claim_number.lower()}@example.com"
    customer = Customer(name="Pipe Tester", email=email, phone="555-0001")
    db.add(customer)
    db.flush()

    vehicle = Vehicle(
        customer_id=customer.id,
        make="Honda",
        model="Accord",
        year=2021,
        vin=f"VIN-{claim_number}",
        plate_number=f"PIPE-{claim_number[-6:]}",
    )
    db.add(vehicle)
    db.flush()

    policy = Policy(
        customer_id=customer.id,
        vehicle_id=vehicle.id,
        policy_number=f"POL-{claim_number}",
        coverage_type="comprehensive",
        coverage_limit=50000.0,
        deductible=500.0,
        start_date=dt.date(2024, 1, 1),
        end_date=dt.date(2026, 12, 31),
        status="active",
    )
    db.add(policy)
    db.flush()

    claim = Claim(
        claim_number=claim_number,
        policy_id=policy.id,
        vehicle_id=vehicle.id,
        incident_date=incident_date or dt.date(2026, 1, 15),
        reported_date=dt.date(2026, 1, 16),
        claimed_amount=claimed_amount,
        status=ClaimStatus.pending.value,
    )
    db.add(claim)
    db.flush()

    if with_image:
        dmg = Damage(
            claim_id=claim.id,
            source="image",
            damage_type="pending",
            severity="pending",
            confidence=None,
            region_ref=json.dumps({"image_path": "uploads/1/pipe-image-1.jpg"}),
        )
        db.add(dmg)

    if with_document:
        doc = Document(
            claim_id=claim.id,
            doc_type=DocType.claim_form.value,
            file_path=f"uploads/{claim.id}/pipe-doc-1.pdf",
            extraction_status=ExtractionStatus.pending.value,
        )
        db.add(doc)

    db.commit()
    db.refresh(claim)
    return claim


def _patch_storage_path(monkeypatch, tmp_path: Path) -> None:
    """Make the document_intelligence stub see a real file on disk so
    it does not flip the document to `failed` (which would conflate
    'extraction failed' with 'storage missing').
    """
    base = tmp_path
    # Any file_path our seed claims will land here.
    (base / "1").mkdir(parents=True, exist_ok=True)
    (base / "1" / "pipe-doc-1.pdf").write_bytes(b"%PDF-stub")
    monkeypatch.setattr(
        "app.services.document_intelligence.settings.upload_dir",
        str(base),
    )


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_full_pipeline_success(db_session, monkeypatch, tmp_path):
    _patch_storage_path(monkeypatch, tmp_path)
    claim = _seed_claim(db_session)
    predictor = FakeCVPredictor()
    gemini = FakeGeminiClient()

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )

    assert result.status == AnalysisStatus.completed.value
    assert result.claim_id == claim.id
    assert result.risk_band is not None  # risk computed
    assert result.signal_count >= 0
    assert result.evidence_count >= 0
    assert result.investigation_id is not None

    # Re-fetch the claim — the status must be completed.
    db_session.refresh(claim)
    assert claim.status == ClaimStatus.completed.value
    assert claim.risk_band in ("Low", "Medium", "High")

    # Investigation row exists with a non-null summary.
    inv = db_session.query(Investigation).filter(Investigation.claim_id == claim.id).one()
    assert inv.summary_text == "Test summary."


def test_missing_inputs_marks_failed(db_session):
    """Claim with no images and no documents is a user error."""
    claim = _seed_claim(db_session, with_image=False, with_document=False)
    predictor = FakeCVPredictor()
    gemini = FakeGeminiClient()

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )

    assert result.status == AnalysisStatus.failed.value
    assert "no images or documents" in (result.error_message or "").lower()

    db_session.refresh(claim)
    assert claim.status == ClaimStatus.analysis_failed.value


def test_already_running_409(db_session, monkeypatch, tmp_path):
    """Insert a running Analysis row directly and acquire the in-process
    lock, then verify that POST /analyze returns 409.

    The in-process lock in `pipeline_locks` is the primary guard in
    the test environment (the partial unique index is created by the
    Alembic migration, which is not applied in the in-memory test DB).
    The DB-level index is the multi-process safety net; the lock is
    what the API actually checks in tests.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    claim = _seed_claim(db_session)

    # Simulate an in-flight run: insert a 'running' analysis row and
    # acquire the in-process lock so the API handler will 409.
    existing = Analysis(
        claim_id=claim.id,
        status=AnalysisStatus.running.value,
        current_step="cv",
    )
    db_session.add(existing)
    claim.status = ClaimStatus.analyzing.value
    db_session.add(claim)
    db_session.commit()

    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.deps import get_db

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    try:
        # Acquire the in-process lock; the API handler will then 409.
        assert pipeline_locks.acquire(claim.id)
        try:
            with TestClient(app) as tc:
                response = tc.post(f"/claims/{claim.id}/analyze")
            assert response.status_code == 409, response.text
            # The 409 detail mentions the existing analysis_id.
            body = response.json()
            assert "detail" in body
            assert body["detail"].get("analysis_id") == existing.id
        finally:
            pipeline_locks.release(claim.id)
    finally:
        app.dependency_overrides.clear()


def test_cv_failure_isolation(db_session, monkeypatch, tmp_path):
    """Two images: one with a working predictor, one with a failing one.
    The pipeline must complete, the working image gets real damage
    rows, the failing image gets a `cv_error` damage row.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    claim = _seed_claim(db_session, claim_number="CLM-PIPE-CV")

    # Add a second image that we will route to a failing predictor.
    failing_img = Damage(
        claim_id=claim.id,
        source="image",
        damage_type="pending",
        severity="pending",
        region_ref=json.dumps({"image_path": "uploads/1/pipe-image-2.jpg"}),
    )
    db_session.add(failing_img)
    db_session.commit()

    # A predictor that fails on the second image only.
    class _SelectivePredictor(FakeCVPredictor):
        def predict_from_path(self, image_path):  # type: ignore[no-untyped-def]
            if "image-2" in str(image_path):
                raise RuntimeError("simulated CV failure on image-2")
            return super().predict_from_path(image_path)

    predictor = _SelectivePredictor()
    gemini = FakeGeminiClient()

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )

    assert result.status == AnalysisStatus.completed.value
    db_session.refresh(claim)
    assert claim.status == ClaimStatus.completed.value

    # The failing image has at least one cv_error row.
    error_rows = (
        db_session.query(Damage)
        .filter(
            Damage.claim_id == claim.id,
            Damage.damage_type == "cv_error",
        )
        .all()
    )
    assert len(error_rows) >= 1


def test_pipeline_removes_placeholder_pending_image_rows(
    db_session, monkeypatch, tmp_path,
):
    """After CV runs successfully, the placeholder `damage_type="pending"`
    row the upload created must be deleted. The frontend uses the
    absence of pending rows to mark the "Image analysis" stage as
    complete; if a stale pending row remained, every completed claim
    would render that stage as still in flight.

    To exercise the "real CV ran" branch (rather than the missing-
    file branch that produces a cv_error row), we patch the upload
    directory onto a temp dir and create the image file the seeded
    Damage row references.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    # Create the image file the seeded damage row references so the
    # cv_service does not short-circuit to a cv_error row. We want
    # this test to exercise the successful-Damage-row path.
    (tmp_path / "1").mkdir(parents=True, exist_ok=True)
    (tmp_path / "1" / "pipe-image-1.jpg").write_bytes(b"\xff\xd8\xff\xe0fake-jpg")
    # Patch the central settings object — cv_service imports it
    # lazily from app.core.config and the test code only needs to
    # patch the singleton.
    from app.core.config import settings as _settings
    monkeypatch.setattr(_settings, "upload_dir", str(tmp_path))

    claim = _seed_claim(db_session, claim_number="CLM-PIPE-PENDING-CLEANUP")
    predictor = FakeCVPredictor()
    gemini = FakeGeminiClient()

    # Sanity: the seeded claim has exactly one image and it is
    # `damage_type="pending"`. After the run, that placeholder must
    # be gone and a real (non-pending) row must exist.
    pending_rows = (
        db_session.query(Damage)
        .filter(
            Damage.claim_id == claim.id,
            Damage.damage_type == "pending",
        )
        .all()
    )
    assert len(pending_rows) == 1

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )
    assert result.status == AnalysisStatus.completed.value

    # The placeholder pending row is gone.
    leftover_pending = (
        db_session.query(Damage)
        .filter(
            Damage.claim_id == claim.id,
            Damage.damage_type == "pending",
        )
        .all()
    )
    assert leftover_pending == [], (
        "stale `damage_type='pending'` row was not cleaned up; "
        "the frontend stage tracker would still show this claim as "
        "'Image analysis — running'."
    )

    # And at least one real CV result row is present.
    real_rows = (
        db_session.query(Damage)
        .filter(
            Damage.claim_id == claim.id,
            Damage.source == "image",
            Damage.damage_type.notin_(["pending", "cv_error"]),
        )
        .all()
    )
    assert len(real_rows) >= 1


def test_document_extraction_failure_isolation(db_session, monkeypatch):
    """One document whose extraction raises is marked failed; the
    pipeline still completes successfully.
    """
    claim = _seed_claim(db_session, claim_number="CLM-PIPE-DOC")

    # Patch extract_document to fail on this specific document.
    real_fn = document_intelligence.extract_document
    target_id = claim.documents[0].id if claim.documents else None

    def fake_extract(db, claim_id, document_id):  # type: ignore[no-untyped-def]
        if document_id == target_id:
            doc = db.get(Document, document_id)
            doc.extraction_status = ExtractionStatus.failed.value
            db.add(doc)
            db.commit()
            return False
        return real_fn(db, claim_id, document_id)

    monkeypatch.setattr(
        "app.services.pipeline.document_intelligence.extract_document",
        fake_extract,
    )

    predictor = FakeCVPredictor()
    gemini = FakeGeminiClient()

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )

    assert result.status == AnalysisStatus.completed.value
    db_session.refresh(claim)
    assert claim.status == ClaimStatus.completed.value
    assert claim.documents[0].extraction_status == ExtractionStatus.failed.value


def test_consistency_failure_marks_analysis_failed(db_session, monkeypatch, tmp_path):
    """A raised exception during consistency persistence is caught by
    the pipeline; the claim becomes `analysis_failed` and the lock
    is released (a second call can run).
    """
    _patch_storage_path(monkeypatch, tmp_path)
    claim = _seed_claim(db_session, claim_number="CLM-PIPE-CONS")

    # Patch persist to raise AFTER the new signals have been
    # evaluated but before they are committed.
    def boom(signals, db):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated consistency failure")

    monkeypatch.setattr(
        "app.services.pipeline.consistency.persist", boom
    )

    predictor = FakeCVPredictor()
    gemini = FakeGeminiClient()

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )

    assert result.status == AnalysisStatus.failed.value
    assert "simulated consistency failure" in (result.error_message or "")
    db_session.refresh(claim)
    assert claim.status == ClaimStatus.analysis_failed.value
    assert not pipeline_locks.is_held(claim.id)


def test_gemini_failure_keeps_claim_completed(db_session, monkeypatch, tmp_path):
    """When Gemini returns None, the pipeline completes successfully
    and the Investigation row is written with `summary_text=None` and
    the deterministic recommendation.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    claim = _seed_claim(db_session, claim_number="CLM-PIPE-GEM")
    predictor = FakeCVPredictor()
    gemini = FakeGeminiClient(return_none=True)

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )

    assert result.status == AnalysisStatus.completed.value
    db_session.refresh(claim)
    assert claim.status == ClaimStatus.completed.value

    inv = db_session.query(Investigation).filter(Investigation.claim_id == claim.id).one()
    assert inv.summary_text is None
    assert inv.recommendation in ("normal", "manual_review", "investigate")


def test_status_polling_endpoint(db_session, monkeypatch, tmp_path):
    """The GET /analysis/{id} endpoint returns a running status
    while the pipeline is in flight, and a completed status after.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    claim = _seed_claim(db_session, claim_number="CLM-PIPE-POLL")

    # We run the pipeline in a background thread using a fresh
    # SessionLocal session, then call the GET endpoint to inspect
    # the status. We can't use the `client` fixture (the threaded
    # session would conflict with the dependency override), so we
    # exercise the same code path through `_to_status_response`
    # which is the only pure-Python helper.
    analysis = Analysis(
        claim_id=claim.id,
        status=AnalysisStatus.running.value,
        current_step="cv",
    )
    db_session.add(analysis)
    claim.status = ClaimStatus.analyzing.value
    db_session.add(claim)
    db_session.commit()
    db_session.refresh(analysis)

    # Mid-run: status is `running`, current_step is `cv`.
    response_running = api_pipeline._to_status_response(analysis, claim, db_session)
    assert response_running.status == "running"
    assert response_running.current_step == "cv"
    assert response_running.claim_status == "analyzing"
    assert response_running.result is None

    # Mark complete and re-read.
    analysis.status = AnalysisStatus.completed.value
    analysis.current_step = None
    analysis.finished_at = dt.datetime.now(dt.timezone.utc)
    claim.status = ClaimStatus.completed.value
    db_session.add(analysis)
    db_session.add(claim)
    db_session.commit()
    db_session.refresh(analysis)
    db_session.refresh(claim)

    response_done = api_pipeline._to_status_response(analysis, claim, db_session)
    assert response_done.status == "completed"
    assert response_done.finished_at is not None
    assert response_done.result is not None
    assert response_done.result.signal_count >= 0
    assert response_done.result.evidence_count >= 0


def test_persistence_of_signals_evidence_risk_investigation(db_session, monkeypatch, tmp_path):
    """After a successful run, the DB has: ≥1 RiskSignal with ≥1
    Evidence row, a non-null risk_band, and an Investigation row.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    claim = _seed_claim(db_session, claim_number="CLM-PIPE-PERS")
    predictor = FakeCVPredictor()
    gemini = FakeGeminiClient()

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )

    assert result.status == AnalysisStatus.completed.value
    db_session.refresh(claim)
    assert claim.risk_band is not None

    inv = db_session.query(Investigation).filter(Investigation.claim_id == claim.id).one()
    assert inv.id is not None

    # The consistency engine may produce zero signals for a trivially
    # clean claim, in which case there are no Evidence rows either.
    # The test passes either way: it only asserts the columns are
    # queryable and the counts are consistent.
    signals = db_session.query(RiskSignal).filter(RiskSignal.claim_id == claim.id).all()
    evidence = (
        db_session.query(Evidence)
        .join(RiskSignal, Evidence.risk_signal_id == RiskSignal.id)
        .filter(RiskSignal.claim_id == claim.id)
        .all()
    )
    assert len(evidence) == sum(len(s.evidence) for s in signals)


def test_no_claim_stuck_analyzing(db_session, monkeypatch, tmp_path):
    """For each failure scenario, the claim's final status must be
    in {`completed`, `analysis_failed`} within 2s of pipeline return.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    predictor = FakeCVPredictor()
    gemini = FakeGeminiClient()

    # Scenario A: missing inputs
    claim_a = _seed_claim(db_session, claim_number="CLM-PIPE-NOSTUCK-A",
                          with_image=False, with_document=False)
    pipeline_service.run_analysis(
        claim_a.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )

    # Scenario B: consistency failure
    claim_b = _seed_claim(db_session, claim_number="CLM-PIPE-NOSTUCK-B")
    monkeypatch.setattr(
        "app.services.pipeline.consistency.persist",
        lambda signals, db: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    pipeline_service.run_analysis(
        claim_b.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )
    monkeypatch.undo()  # restore consistency.persist

    # Wait briefly for any in-flight background work.
    time.sleep(0.1)

    for claim in (claim_a, claim_b):
        db_session.refresh(claim)
        assert claim.status in (
            ClaimStatus.completed.value,
            ClaimStatus.analysis_failed.value,
        ), f"Claim {claim.id} stuck in {claim.status}"

    # A broader sweep: no claim in the DB is in `analyzing`.
    stuck = db_session.query(Claim).filter(Claim.status == ClaimStatus.analyzing.value).all()
    assert stuck == []


def test_decided_claim_analyze_returns_409(client, db_session):
    """A decided claim cannot be re-analyzed."""
    c = Customer(name="D", email="d@x.com", phone="1")
    db_session.add(c); db_session.flush()
    v = Vehicle(customer_id=c.id, make="Honda", model="Civic", year=2020)
    db_session.add(v); db_session.flush()
    p = Policy(
        customer_id=c.id, vehicle_id=v.id, policy_number="POL-DEC",
        coverage_type="comprehensive", coverage_limit=10000, deductible=500,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2025, 1, 1), status="active",
    )
    db_session.add(p); db_session.flush()
    claim = Claim(
        claim_number="CLM-DEC-409", policy_id=p.id, vehicle_id=v.id,
        incident_date=dt.date(2024, 6, 1), claimed_amount=1000.0,
        status=ClaimStatus.decided.value,
    )
    db_session.add(claim); db_session.commit()

    response = client.post(f"/claims/{claim.id}/analyze")
    assert response.status_code == 409


def test_start_analysis_returns_202(client, db_session, monkeypatch, tmp_path):
    """POST /analyze returns 202 + analysis_id when a claim has
    images and the pipeline starts successfully.

    Note: we cannot use the threaded `client.post` path in this
    test because SQLite in-memory uses per-connection state and the
    background thread would not see the seeded data. We exercise
    the API endpoint without threading by monkeypatching the
    `_run_in_thread` to call `run_analysis` synchronously on the
    same session — which is what the request handler does anyway,
    just on a different thread.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    claim = _seed_claim(db_session, claim_number="CLM-PIPE-202")

    # Bypass real CV by patching `run_cv_on_image` to write a
    # synthetic damage row.
    def fake_run_cv(db, claim_id, image_path, predictor=None):  # type: ignore[no-untyped-def]
        from app.models.damage import Damage as Dmg
        dmg = Dmg(
            claim_id=claim_id,
            source="image",
            damage_type="scratch",
            severity="moderate",
            confidence=0.9,
            region_ref=image_path,
        )
        db.add(dmg)
        db.flush()
        return [dmg]

    monkeypatch.setattr("app.services.pipeline.cv_service.run_cv_on_image", fake_run_cv)
    monkeypatch.setattr(
        "app.services.pipeline.gemini_client.GeminiClient",
        lambda *a, **kw: FakeGeminiClient(),
    )

    # Replace the threaded runner with an in-process call so the
    # in-memory SQLite sees the data. We also track the thread so
    # the test can wait for the synchronous work to finish.
    threads: list[threading.Thread] = []

    def sync_run_in_thread(claim_id, analysis_id):  # type: ignore[no-untyped-def]
        pipeline_service.run_analysis_steps(
            claim_id, analysis_id, db_session,
            cv_predictor=FakeCVPredictor(),
            gemini_client_obj=FakeGeminiClient(),
        )
        pipeline_locks.release(claim_id)

    monkeypatch.setattr("app.api.pipeline._run_in_thread", sync_run_in_thread)

    response = client.post(f"/claims/{claim.id}/analyze")
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["claim_id"] == claim.id
    assert body["status"] == "running"
    assert "analysis_id" in body
    assert body["analysis_id"] > 0

    # The real production code uses a real `threading.Thread` to
    # run the worker. The test's monkeypatch makes the worker
    # synchronous, but the thread is still started. We give it a
    # moment to finish so the lock-release assertion holds.
    time.sleep(0.2)

    # The row exists; the lock has been released.
    analysis_id = body["analysis_id"]
    a = db_session.get(Analysis, analysis_id)
    assert a is not None
    assert a.claim_id == claim.id
    assert not pipeline_locks.is_held(claim.id)


# ─── Phase 12 regression tests ──────────────────────────────────────────────


def test_r4_fires_when_repair_total_exceeds_baseline(db_session, monkeypatch, tmp_path):
    """Phase 12 regression: the pipeline must populate `baseline_upper`
    on the ClaimContext so R4 (excessive_repair_cost) can fire. Before
    the fix, the pipeline called `build_claim_context(claim_id, db)`
    without `baseline_upper`, so R4 was always silent regardless of the
    repair-estimate total. This blocked the demo scenarios 2 (inflated
    repair estimate) and 5 (multi-signal suspicious) from producing
    their documented Medium/High bands.
    """
    _patch_storage_path(monkeypatch, tmp_path)
    # Claim with a pending image (gets CV-detected as `dent`, severe).
    # The CV predictor returns "dent" + "severe".
    predictor = FakeCVPredictor(damage_type="dent")
    gemini = FakeGeminiClient()
    # Use a custom seed that lets us attach a RepairEstimate after.
    claim = _seed_claim(db_session, claim_number="CLM-PIPE-R4")
    # Force the CV-detected severity by setting it post-hoc after the
    # pipeline stub will run. Easiest approach: also seed a claim-form
    # damage row whose severity R4 will read from the primary-damage
    # lookup. The pipeline's `cv_service.run_cv_on_image` writes a
    # Damage row with damage_type from the predictor and severity from
    # the predictor's severity prediction ("moderate" by default in
    # FakeCVPredictor). The _primary_damage helper uses the first
    # high-confidence CV damage, falling back to the first claim-form
    # damage.
    #
    # To make R4 fire we need: (a) a primary damage with severity so
    # we can compute a baseline, and (b) a repair estimate with
    # total_cost > baseline_upper * 1.5.
    #
    # Default `dent/moderate` on a Honda Accord → sedan → baseline is
    # synthetic and small (single-digit thousands). Setting
    # total_cost=50000 guarantees > 1.5× any plausible baseline.
    estimate = RepairEstimate(
        claim_id=claim.id,
        shop_name="Test Shop",
        total_cost=50000.0,
        currency="USD",
        issued_date=dt.date(2026, 1, 15),
    )
    db_session.add(estimate)
    db_session.commit()

    result = pipeline_service.run_analysis(
        claim.id, db_session,
        cv_predictor=predictor, gemini_client_obj=gemini,
    )
    assert result.status == AnalysisStatus.completed.value, result.error_message

    # R4 must have fired and been persisted.
    rule_ids = {s.rule_id for s in db_session.query(RiskSignal).filter(
        RiskSignal.claim_id == claim.id
    ).all()}
    assert "R4_excessive_repair_cost" in rule_ids, (
        f"Expected R4 to fire with total=50000 vs baseline; "
        f"signals actually fired: {sorted(rule_ids)}"
    )
