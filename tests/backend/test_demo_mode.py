"""
Phase 13 — Demo mode tests for the CV service and Gemini client.

When `USE_DEMO_CV=1` (or `settings.use_demo_cv=True`) the CV service
must use `_DemoCVPredictor` instead of trying to load the trained
model. When `USE_DEMO_GEMINI=1` the Gemini client must return a
deterministic stub built from the input. These tests pin both paths
so a future change does not silently break the demo experience.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.models.customer import Customer
from app.models.policy import Policy
from app.models.vehicle import Vehicle
from app.models.claim import Claim
from app.models.damage import Damage

from app.services import cv_service


@pytest.fixture
def demo_cv_settings():
    """Force the cached `Settings` singleton into demo mode for the
    duration of the test.

    `pydantic-settings` reads env vars on `__init__`; the cached
    instance is built once at first import. We toggle the relevant
    fields directly and restore them after the test, regardless of
    env-var sequencing.
    """
    from app.core.config import settings
    saved_cv = settings.use_demo_cv
    saved_gem = settings.use_demo_gemini
    settings.use_demo_cv = True
    settings.use_demo_gemini = True
    try:
        yield
    finally:
        settings.use_demo_cv = saved_cv
        settings.use_demo_gemini = saved_gem


def _make_claim_with_image(db_session, *, image_filename: str):
    cust = Customer(name="Demo CV", email="demo_cv@test.com", phone="0")
    db_session.add(cust); db_session.flush()
    veh = Vehicle(customer_id=cust.id, make="Honda", model="Civic",
                  year=2020, vin="DEMOCVVIN", plate_number="DEMO-1")
    db_session.add(veh); db_session.flush()
    pol = Policy(customer_id=cust.id, vehicle_id=veh.id,
                 policy_number="DEMOCVPOL", coverage_type="comprehensive",
                 coverage_limit=50000, deductible=500,
                 start_date=__import__("datetime").date(2025, 1, 1),
                 end_date=__import__("datetime").date(2026, 12, 31),
                 status="active")
    db_session.add(pol); db_session.flush()
    claim = Claim(claim_number="DEMO-CV-CLM", policy_id=pol.id,
                  vehicle_id=veh.id,
                  incident_date=__import__("datetime").date(2026, 2, 15),
                  reported_date=__import__("datetime").date(2026, 2, 16),
                  claimed_amount=1000.0, status="pending")
    db_session.add(claim); db_session.flush()
    region_ref = json.dumps({"image_path": f"uploads/{claim.id}/{image_filename}"})
    dmg = Damage(claim_id=claim.id, source="image",
                 damage_type="pending", severity="pending", confidence=None,
                 region_ref=region_ref)
    db_session.add(dmg); db_session.flush()
    # Place a placeholder file.
    from app.core.config import settings
    import os
    base = Path(settings.upload_dir)
    if not base.is_absolute():
        base = Path(os.getcwd()) / base
    claim_dir = base / str(claim.id)
    claim_dir.mkdir(parents=True, exist_ok=True)
    (claim_dir / image_filename).write_bytes(
        b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9"
    )
    return claim, dmg


def test_demo_cv_predicts_scratch_for_scratch_filename(
    db_session, demo_cv_settings
):
    claim, dmg = _make_claim_with_image(db_session, image_filename="rear-scratch.jpg")
    rows = cv_service.run_cv_on_image(
        db_session, claim.id, dmg.region_ref, predictor=None,
    )
    assert len(rows) == 1
    assert rows[0].damage_type == "scratch"
    assert rows[0].severity == "minor"
    assert rows[0].confidence is not None and rows[0].confidence > 0.5


def test_demo_cv_predicts_dent_for_dent_filename(
    db_session, demo_cv_settings
):
    claim, dmg = _make_claim_with_image(db_session, image_filename="small-dent.jpg")
    rows = cv_service.run_cv_on_image(
        db_session, claim.id, dmg.region_ref, predictor=None,
    )
    assert len(rows) == 1
    assert rows[0].damage_type == "dent"
    assert rows[0].severity == "minor"


def test_demo_cv_predicts_bumper_for_bumper_filename(
    db_session, demo_cv_settings
):
    claim, dmg = _make_claim_with_image(db_session, image_filename="bumper.jpg")
    rows = cv_service.run_cv_on_image(
        db_session, claim.id, dmg.region_ref, predictor=None,
    )
    assert len(rows) == 1
    assert rows[0].damage_type == "bumper_damage"
    assert rows[0].severity == "moderate"


def test_demo_cv_falls_back_to_scratch_for_unknown_filename(
    db_session, demo_cv_settings
):
    claim, dmg = _make_claim_with_image(db_session, image_filename="random_photo.jpg")
    rows = cv_service.run_cv_on_image(
        db_session, claim.id, dmg.region_ref, predictor=None,
    )
    assert len(rows) == 1
    assert rows[0].damage_type == "scratch"
    assert rows[0].severity == "minor"


def test_demo_gemini_returns_approve_when_no_signals():
    from app.services.gemini_client import GeminiClient, InvestigationInput
    from app.core import config as config_mod
    config_mod.get_settings.cache_clear()
    client = GeminiClient(api_key="ignored")
    out = client._generate_demo(InvestigationInput(
        claim_id=1,
        risk_score=0.0,
        risk_band="Low",
        risk_signals=[],
        evidence=(),
        extracted_documents_summary={},
        cv_findings=(),
    ))
    assert out.recommendation == "normal"
    assert out.model_version == "demo_deterministic_v1"


def test_demo_gemini_returns_manual_review_on_high_signals():
    from app.services.gemini_client import GeminiClient, InvestigationInput
    client = GeminiClient(api_key="ignored")
    out = client._generate_demo(InvestigationInput(
        claim_id=1,
        risk_score=70.0,
        risk_band="High",
        risk_signals=(
            {"rule_id": "R1_unsupported_damage", "description": "x",
             "severity": "high", "category": "image_claim_consistency"},
        ),
        evidence=(),
        extracted_documents_summary={},
        cv_findings=(),
    ))
    assert out.recommendation == "manual_review"
    assert out.summary  # non-empty


def test_demo_gemini_returns_more_info_on_medium_only():
    from app.services.gemini_client import GeminiClient, InvestigationInput
    client = GeminiClient(api_key="ignored")
    out = client._generate_demo(InvestigationInput(
        claim_id=1,
        risk_score=40.0,
        risk_band="Medium",
        risk_signals=(
            {"rule_id": "R2_severity_mismatch", "description": "x",
             "severity": "medium", "category": "claim_description_consistency"},
        ),
        evidence=(),
        extracted_documents_summary={},
        cv_findings=(),
    ))
    assert out.recommendation == "investigate"
