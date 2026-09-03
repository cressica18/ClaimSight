"""
Demo data generation script — Phase 13.

This script seeds the database with the 5 synthetic demo scenarios from
Section 3.3 of the implementation blueprint:

  1. Legitimate claim (expected risk: Low)
  2. Inflated repair estimate (expected risk: Medium–High)
  3. Image/document mismatch (expected risk: High)
  4. Previous-claim overlap (expected risk: Medium–High)
  5. Multi-signal suspicious claim (expected risk: High)

The script is deterministic (no randomness; all data is hand-crafted)
and idempotent: re-running it is a no-op if the customers already
exist. The script creates the claim graph (customer, vehicle, policy,
accident, images, documents, repair estimates, previous claims) for
each scenario. It does NOT run the analysis pipeline — by default the
claims are left in `status=pending` so the demo user can click
"Start analysis" and watch the pipeline progress. Pass `--analyze`
to run the pipeline immediately for each claim (uses fake CV and
Gemini to keep the run fully offline).

Usage (from the project root, with the backend's venv active):
    cd backend
    python ../scripts/generate_demo_data.py
    python ../scripts/generate_demo_data.py --analyze
    python ../scripts/generate_demo_data.py --reset   # wipe & reseed

The script reads DATABASE_URL from the backend's .env via
`app.core.config.settings`. It expects the schema to be present
(run `alembic upgrade head` first if you have not yet).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

# Add the backend directory to sys.path so `app.*` imports work when
# this script is invoked from the project root.
_BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Also add the project root so the FakeCV in `_analyze_all_claims`
# can import `ml.inference.predictor` (for the typed CVPrediction
# return value). The real `ml` package is at the repo root, not the
# backend directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Force the working directory to backend so `settings.upload_dir` (a
# relative path) resolves consistently with the running FastAPI app.
os.chdir(str(_BACKEND_DIR))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.models.accident import Accident  # noqa: E402
from app.models.claim import Claim  # noqa: E402
from app.models.customer import Customer  # noqa: E402
from app.models.damage import Damage  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.enums import (  # noqa: E402
    ClaimStatus,
    DocType,
    ExtractionStatus,
)
from app.models.policy import Policy  # noqa: E402
from app.models.previous_claim import PreviousClaim  # noqa: E402
from app.models.repair import RepairEstimate, RepairItem  # noqa: E402
from app.models.vehicle import Vehicle  # noqa: E402

RANDOM_SEED = 42

# All claims are anchored to this calendar date so the demo is
# reproducible: incident_date is consistent, policy is active, and
# no time-based "near boundary" rule fires by accident.
_BASE_INCIDENT = dt.date(2026, 2, 15)
_BASE_REPORTED = dt.date(2026, 2, 16)
_POLICY_START = dt.date(2025, 1, 1)
_POLICY_END = dt.date(2026, 12, 31)

# A minimal but valid PDF — a single blank page. The earlier
# `%PDF-1.4\n% demo stub\n` 21-byte stub was missing the xref
# table and `%%EOF` marker, so Chrome's built-in PDF viewer
# refused it with "Failed to load PDF document". This byte
# string is exactly 325 bytes and opens cleanly in every modern
# PDF viewer. It contains no invented text content.
_MINIMAL_PDF_BYTES = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>\nendobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n"
    b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n183\n%%EOF\n"
)


# ─── DB plumbing ───────────────────────────────────────────────────────────


def _ensure_session() -> Session:
    """Open a SessionLocal; the caller is responsible for closing it."""
    return SessionLocal()


def _drop_demo_data(db: Session) -> None:
    """Delete all demo data. Demo customers are looked up by email and
    removed in dependency order: claims → accidents → documents →
    damages → repair estimates → repair items → previous claims →
    policies → vehicles → customers. The DB-level `ondelete="RESTRICT"`
    on policies.customer_id / policies.vehicle_id prevents the
    convenient "delete the customer, let it cascade" path; we delete
    bottom-up explicitly.
    """
    from app.models.accident import Accident
    from app.models.damage import Damage
    from app.models.document import Document
    from app.models.previous_claim import PreviousClaim
    from app.models.repair import RepairEstimate, RepairItem
    from app.models.risk_signal import RiskSignal
    from app.models.analysis import Analysis
    from app.models.evidence import Evidence
    from app.models.policy import Policy
    from app.models.vehicle import Vehicle

    demo_emails = [c["email"] for c in _SCENARIOS]
    customers = db.execute(
        select(Customer).where(Customer.email.in_(demo_emails))
    ).scalars().all()
    if not customers:
        return
    cust_ids = [c.id for c in customers]

    # Bottom-up deletion: remove child rows before their parents.
    claims = db.execute(
        select(Claim).where(Claim.policy_id.in_(
            select(Policy.id).where(Policy.customer_id.in_(cust_ids))
        ))
    ).scalars().all()
    claim_ids = [c.id for c in claims]

    if claim_ids:
        db.query(RepairItem).filter(
            RepairItem.repair_estimate_id.in_(
                select(RepairEstimate.id).where(
                    RepairEstimate.claim_id.in_(claim_ids)
                )
            )
        ).delete(synchronize_session=False)
        db.query(RepairEstimate).filter(
            RepairEstimate.claim_id.in_(claim_ids)
        ).delete(synchronize_session=False)
        db.query(Damage).filter(
            Damage.claim_id.in_(claim_ids)
        ).delete(synchronize_session=False)
        db.query(Document).filter(
            Document.claim_id.in_(claim_ids)
        ).delete(synchronize_session=False)
        db.query(Accident).filter(
            Accident.claim_id.in_(claim_ids)
        ).delete(synchronize_session=False)
        # Evidence hangs off RiskSignal, not Claim directly.
        db.query(Evidence).filter(
            Evidence.risk_signal_id.in_(
                select(RiskSignal.id).where(
                    RiskSignal.claim_id.in_(claim_ids)
                )
            )
        ).delete(synchronize_session=False)
        db.query(RiskSignal).filter(
            RiskSignal.claim_id.in_(claim_ids)
        ).delete(synchronize_session=False)
        db.query(Analysis).filter(
            Analysis.claim_id.in_(claim_ids)
        ).delete(synchronize_session=False)
        db.query(Claim).filter(Claim.id.in_(claim_ids)).delete(
            synchronize_session=False,
        )

    db.query(PreviousClaim).filter(
        PreviousClaim.customer_id.in_(cust_ids)
    ).delete(synchronize_session=False)
    db.query(Policy).filter(
        Policy.customer_id.in_(cust_ids)
    ).delete(synchronize_session=False)
    db.query(Vehicle).filter(
        Vehicle.customer_id.in_(cust_ids)
    ).delete(synchronize_session=False)
    db.query(Customer).filter(
        Customer.id.in_(cust_ids)
    ).delete(synchronize_session=False)
    db.commit()


# ─── Scenario seeders ──────────────────────────────────────────────────────


def _seed_base(
    db: Session,
    *,
    scenario: str,
    email: str,
    customer_name: str,
    vin: str,
    plate: str,
    make: str = "Honda",
    model: str = "Accord",
    year: int = 2021,
    policy_number: str | None = None,
    coverage_type: str = "comprehensive",
) -> tuple[Customer, Vehicle, Policy]:
    """Create a customer + vehicle + policy for a scenario. Returns
    the three freshly-flushed objects. Idempotent on email.
    """
    existing = db.execute(
        select(Customer).where(Customer.email == email)
    ).scalars().first()
    if existing is not None:
        cust = existing
        vehicle = db.execute(
            select(Vehicle).where(Vehicle.customer_id == cust.id)
        ).scalars().first()
        policy = db.execute(
            select(Policy).where(Policy.customer_id == cust.id)
        ).scalars().first()
        assert vehicle is not None and policy is not None
        return cust, vehicle, policy

    cust = Customer(name=customer_name, email=email, phone="555-0100")
    db.add(cust); db.flush()

    vehicle = Vehicle(
        customer_id=cust.id, make=make, model=model, year=year,
        vin=vin, plate_number=plate,
    )
    db.add(vehicle); db.flush()

    policy = Policy(
        customer_id=cust.id, vehicle_id=vehicle.id,
        policy_number=policy_number or f"POL-{scenario}-{vin[-6:]}",
        coverage_type=coverage_type,
        coverage_limit=50000.0, deductible=500.0,
        start_date=_POLICY_START, end_date=_POLICY_END,
        status="active",
    )
    db.add(policy); db.flush()
    return cust, vehicle, policy


def _make_claim(
    db: Session,
    *,
    policy: Policy,
    vehicle: Vehicle,
    claim_number: str,
    claimed_amount: float,
    incident_date: dt.date = _BASE_INCIDENT,
    reported_date: dt.date = _BASE_REPORTED,
) -> Claim:
    existing = db.execute(
        select(Claim).where(Claim.claim_number == claim_number)
    ).scalars().first()
    if existing is not None:
        return existing
    claim = Claim(
        claim_number=claim_number,
        policy_id=policy.id, vehicle_id=vehicle.id,
        incident_date=incident_date, reported_date=reported_date,
        claimed_amount=claimed_amount,
        status=ClaimStatus.pending.value,
    )
    db.add(claim); db.flush()
    return claim


def _add_image(
    db: Session, claim: Claim, *, image_basename: str,
    damage_type: str = "pending", severity: str = "pending",
) -> Damage:
    """Add a pending image Damage row. The pipeline runs CV on it
    when the user clicks "Start analysis". Idempotent on
    `region_ref.image_path` so re-runs do not duplicate rows.
    """
    region_ref = json.dumps({
        "image_path": f"uploads/{claim.id}/{image_basename}",
    })
    existing = db.execute(
        select(Damage).where(
            Damage.claim_id == claim.id,
            Damage.region_ref == region_ref,
        )
    ).scalars().first()
    if existing is not None:
        return existing
    dmg = Damage(
        claim_id=claim.id, source="image",
        damage_type=damage_type, severity=severity, confidence=None,
        region_ref=region_ref,
    )
    db.add(dmg); db.flush()
    return dmg


def _add_document(
    db: Session, claim: Claim, *,
    doc_type: str, basename: str,
    extraction_status: str = ExtractionStatus.pending.value,
) -> Document:
    file_path = f"uploads/{claim.id}/{basename}"
    existing = db.execute(
        select(Document).where(Document.file_path == file_path)
    ).scalars().first()
    if existing is not None:
        return existing
    doc = Document(
        claim_id=claim.id, doc_type=doc_type,
        file_path=file_path,
        extraction_status=extraction_status,
    )
    db.add(doc); db.flush()
    return doc


def _add_accident(
    db: Session, claim: Claim, *,
    description: str, location: str, incident_type: str,
) -> Accident | None:
    """Idempotent on claim_id — one accident per claim."""
    existing = db.execute(
        select(Accident).where(Accident.claim_id == claim.id)
    ).scalars().first()
    if existing is not None:
        return existing
    acc = Accident(
        claim_id=claim.id, description=description,
        location=location, incident_type=incident_type,
    )
    db.add(acc); db.flush()
    return acc


def _add_claim_form_damage(
    db: Session, claim: Claim, *,
    damage_type: str, severity: str, confidence: float = 0.95,
) -> Damage:
    """Idempotent — find the claim_form row for the (claim, type) pair
    before adding. The first scenario (legitimate) does not add a
    claim_form Damage row (it has only the image Damage + the
    pipeline-detected CV row); subsequent re-runs will not duplicate
    because we look up by (claim_id, source, damage_type).
    """
    existing = db.execute(
        select(Damage).where(
            Damage.claim_id == claim.id,
            Damage.source == "claim_form",
            Damage.damage_type == damage_type,
        )
    ).scalars().first()
    if existing is not None:
        return existing
    dmg = Damage(
        claim_id=claim.id, source="claim_form",
        damage_type=damage_type, severity=severity, confidence=confidence,
    )
    db.add(dmg); db.flush()
    return dmg


def _add_repair_estimate(
    db: Session, claim: Claim, *,
    shop_name: str, total_cost: float,
    items: list[dict],
) -> RepairEstimate:
    """Idempotent on (claim_id, shop_name)."""
    existing = db.execute(
        select(RepairEstimate).where(
            RepairEstimate.claim_id == claim.id,
            RepairEstimate.shop_name == shop_name,
        )
    ).scalars().first()
    if existing is not None:
        return existing
    est = RepairEstimate(
        claim_id=claim.id, shop_name=shop_name,
        total_cost=total_cost, currency="USD", issued_date=_BASE_INCIDENT,
    )
    db.add(est); db.flush()
    for it in items:
        db.add(RepairItem(
            repair_estimate_id=est.id,
            part_name=it["part_name"],
            operation=it["operation"],
            cost=it["cost"],
            labor_hours=it["labor_hours"],
        ))
    db.flush()
    return est


def _seed_placeholder_files(
    db: Session, claim: Claim, basenames: list[str]
) -> None:
    """Write small placeholder files to disk so the CV service and
    document-intelligence stub can find them. The CV service needs
    a real file to succeed; the document stub flips to `failed` if
    the file is missing.
    """
    base = Path(settings.upload_dir)
    if not base.is_absolute():
        base = Path(os.getcwd()) / base
    claim_dir = base / str(claim.id)
    claim_dir.mkdir(parents=True, exist_ok=True)
    for name in basenames:
        path = claim_dir / name
        if not path.exists():
            # Minimal JPEG / PDF headers so file-type detection does
            # not crash anything. The actual content does not matter
            # because CV is mocked during demo runs.
            if name.lower().endswith((".jpg", ".jpeg")):
                path.write_bytes(
                    b"\xff\xd8\xff\xe0" + b"\x00" * 32 + b"\xff\xd9"
                )
            else:
                # A minimal but valid PDF that Chrome's built-in
                # viewer will open. The old 21-byte stub was missing
                # the xref table + trailer so the browser refused
                # it with "Failed to load PDF document". This is a
                # blank single-page document — we are not inventing
                # any content for a real claim.
                path.write_bytes(_MINIMAL_PDF_BYTES)


# ─── Scenario 1 — Legitimate ────────────────────────────────────────────────


def seed_s1_legitimate(db: Session) -> Claim:
    """Legitimate scratch on a Honda Accord. CV-detected as `scratch` /
    `minor`; claim-form matches; small repair estimate; no prior
    history. Expected: Low band.
    """
    cust, veh, pol = _seed_base(
        db, scenario="S1", email="demo-s1-legit@example.com",
        customer_name="Alice Legitimate", vin="VIN-DEMO-S1-LEGIT",
        plate="DEMO-S1-LEG",
    )
    claim = _make_claim(
        db, policy=pol, vehicle=veh, claim_number="CLM-DEMO-S1-LEGIT",
        claimed_amount=600.0,
    )
    _add_image(db, claim, image_basename="rear-scratch.jpg")
    _add_document(db, claim, doc_type=DocType.claim_form.value,
                  basename="claim-form.pdf")
    _add_document(db, claim, doc_type=DocType.estimate.value,
                  basename="repair-estimate.pdf")
    # Modest repair estimate — well within baseline.
    _add_repair_estimate(
        db, claim, shop_name="Accurate Auto Body", total_cost=600.0,
        items=[
            {"part_name": "rear quarter panel", "operation": "repair",
             "cost": 400.0, "labor_hours": 2.0},
            {"part_name": "paint", "operation": "paint",
             "cost": 200.0, "labor_hours": 2.0},
        ],
    )
    _seed_placeholder_files(db, claim, [
        "rear-scratch.jpg", "claim-form.pdf", "repair-estimate.pdf",
    ])
    return claim


# ─── Scenario 2 — Inflated repair estimate ──────────────────────────────────


def seed_s2_inflated(db: Session) -> Claim:
    """Small dent, but the repair estimate is wildly inflated. R4
    should fire. Expected: Medium–High band.
    """
    cust, veh, pol = _seed_base(
        db, scenario="S2", email="demo-s2-inflated@example.com",
        customer_name="Bob Inflated", vin="VIN-DEMO-S2-INFL",
        plate="DEMO-S2-INF",
    )
    claim = _make_claim(
        db, policy=pol, vehicle=veh, claim_number="CLM-DEMO-S2-INFLATED",
        claimed_amount=50000.0,
    )
    _add_image(db, claim, image_basename="small-dent.jpg")
    _add_document(db, claim, doc_type=DocType.claim_form.value,
                  basename="claim-form.pdf")
    _add_document(db, claim, doc_type=DocType.estimate.value,
                  basename="inflated-estimate.pdf")
    _add_repair_estimate(
        db, claim, shop_name="Premium Auto Spa", total_cost=50000.0,
        items=[
            {"part_name": "front bumper", "operation": "replace",
             "cost": 50000.0, "labor_hours": 10.0},
        ],
    )
    _seed_placeholder_files(db, claim, [
        "small-dent.jpg", "claim-form.pdf", "inflated-estimate.pdf",
    ])
    return claim


# ─── Scenario 3 — Image/document mismatch ───────────────────────────────────


def seed_s3_mismatch(db: Session) -> Claim:
    """Image shows a dent; claim form lists a headlight damage (R1).
    Description says "completely totaled" but CV says minor (R2).
    Policy is third-party and does not cover own-vehicle damage (R6).
    Expected: High band.
    """
    cust, veh, pol = _seed_base(
        db, scenario="S3", email="demo-s3-mismatch@example.com",
        customer_name="Carol Mismatch", vin="VIN-DEMO-S3-MISM",
        plate="DEMO-S3-MIS", coverage_type="third_party",
    )
    claim = _make_claim(
        db, policy=pol, vehicle=veh, claim_number="CLM-DEMO-S3-MISMATCH",
        claimed_amount=15000.0,
    )
    _add_image(db, claim, image_basename="bumper-dent.jpg")
    _add_document(db, claim, doc_type=DocType.claim_form.value,
                  basename="claim-form.pdf")
    _add_claim_form_damage(
        db, claim, damage_type="headlight_damage", severity="minor",
    )
    _add_accident(
        db, claim,
        description=("Vehicle was completely totaled; every panel destroyed "
                     "in the collision."),
        location="Highway 101", incident_type="collision",
    )
    _seed_placeholder_files(db, claim, [
        "bumper-dent.jpg", "claim-form.pdf",
    ])
    return claim


# ─── Scenario 4 — Previous-claim overlap ────────────────────────────────────


def seed_s4_prev_overlap(db: Session) -> Claim:
    """A prior claim for the same vehicle in the 6-month window with
    overlapping damage text. R5 should fire. Expected: Medium–High.
    """
    cust, veh, pol = _seed_base(
        db, scenario="S4", email="demo-s4-prev@example.com",
        customer_name="Dave Repeater", vin="VIN-DEMO-S4-PREV",
        plate="DEMO-S4-PRV",
    )
    claim = _make_claim(
        db, policy=pol, vehicle=veh, claim_number="CLM-DEMO-S4-PREV",
        claimed_amount=2200.0,
    )
    _add_image(db, claim, image_basename="bumper.jpg")
    _add_document(db, claim, doc_type=DocType.claim_form.value,
                  basename="claim-form.pdf")
    # Previous claim for the same vehicle 3 months earlier, with
    # overlapping damage text so the Jaccard overlap (R5) triggers.
    pr_existing = db.execute(
        select(PreviousClaim).where(PreviousClaim.claim_number == "PRV-DEMO-S4-1")
    ).scalars().first()
    if pr_existing is None:
        db.add(PreviousClaim(
            customer_id=cust.id, vehicle_id=veh.id,
            claim_number="PRV-DEMO-S4-1",
            incident_date=dt.date(2025, 11, 10),
            damage_summary=("Rear bumper panel damage from a parking "
                            "collision. The bumper was dented and the "
                            "quarter panel was scratched."),
            claimed_amount=2100.0,
        ))
    _add_claim_form_damage(
        db, claim, damage_type="bumper_damage", severity="moderate",
    )
    _seed_placeholder_files(db, claim, [
        "bumper.jpg", "claim-form.pdf",
    ])
    return claim


# ─── Scenario 5 — Multi-signal suspicious ────────────────────────────────────


def seed_s5_multi_signal(db: Session) -> Claim:
    """Stacks R1 (unsupported damage), R2 (severity mismatch), and
    R4 (inflated estimate) on the same claim. Expected: High band
    with ≥3 signals.
    """
    cust, veh, pol = _seed_base(
        db, scenario="S5", email="demo-s5-multi@example.com",
        customer_name="Eve Multi-Signal", vin="VIN-DEMO-S5-MULTI",
        plate="DEMO-S5-MUL",
    )
    claim = _make_claim(
        db, policy=pol, vehicle=veh, claim_number="CLM-DEMO-S5-MULTI",
        claimed_amount=80000.0,
    )
    _add_image(db, claim, image_basename="front-damage.jpg")
    _add_document(db, claim, doc_type=DocType.claim_form.value,
                  basename="claim-form.pdf")
    _add_document(db, claim, doc_type=DocType.estimate.value,
                  basename="inflated-estimate.pdf")
    _add_claim_form_damage(
        db, claim, damage_type="headlight_damage", severity="minor",
    )
    _add_accident(
        db, claim,
        description=("Vehicle was completely totaled; every panel "
                     "destroyed in the highway collision."),
        location="Interstate 5", incident_type="collision",
    )
    _add_repair_estimate(
        db, claim, shop_name="Luxury Auto Restoration", total_cost=80000.0,
        items=[
            {"part_name": "front bumper", "operation": "replace",
             "cost": 80000.0, "labor_hours": 15.0},
        ],
    )
    _seed_placeholder_files(db, claim, [
        "front-damage.jpg", "claim-form.pdf", "inflated-estimate.pdf",
    ])
    return claim


# ─── Registry ───────────────────────────────────────────────────────────────


_SCENARIOS = [
    {"key": "S1", "name": "Legitimate", "email": "demo-s1-legit@example.com",
     "claim_number": "CLM-DEMO-S1-LEGIT", "seeder": seed_s1_legitimate},
    {"key": "S2", "name": "Inflated repair estimate",
     "email": "demo-s2-inflated@example.com",
     "claim_number": "CLM-DEMO-S2-INFLATED", "seeder": seed_s2_inflated},
    {"key": "S3", "name": "Image / document mismatch",
     "email": "demo-s3-mismatch@example.com",
     "claim_number": "CLM-DEMO-S3-MISMATCH", "seeder": seed_s3_mismatch},
    {"key": "S4", "name": "Previous-claim overlap",
     "email": "demo-s4-prev@example.com",
     "claim_number": "CLM-DEMO-S4-PREV", "seeder": seed_s4_prev_overlap},
    {"key": "S5", "name": "Multi-signal suspicious",
     "email": "demo-s5-multi@example.com",
     "claim_number": "CLM-DEMO-S5-MULTI", "seeder": seed_s5_multi_signal},
]


# ─── Optional: run the pipeline for every seeded claim ─────────────────────


def _analyze_all_claims(db: Session, claims: list[Claim]) -> dict[int, dict]:
    """Run the pipeline for every supplied claim using fake CV and
    fake Gemini. Returns a {claim_id: result_summary} map.

    Imported lazily so the script can still seed data when the
    Phase 11 pipeline module is unavailable (e.g. mid-Phase-12).
    """
    from app.services import pipeline as pipeline_service

    class _FakeCV:
        def __init__(self, damage_type: str, severity: str):
            self.damage_type = damage_type
            self.severity = severity

        def predict_from_path(self, image_path):  # noqa: D401
            from ml.inference.predictor import (
                CVPrediction, DamageTypePrediction, SeverityPrediction,
            )
            return CVPrediction(
                damage_types=[DamageTypePrediction(
                    label=self.damage_type, confidence=0.92,
                )],
                severity=SeverityPrediction(
                    label=self.severity, confidence=0.88,
                ),
                low_confidence=False,
                model_version="demo_fake_cv_v1",
                source_image=str(image_path),
                timestamp="2026-02-15T00:00:00",
                error=None,
            )

    class _FakeGemini:
        def generate(self, input):  # noqa: D401
            from app.services.gemini_client import InvestigationOutput
            return InvestigationOutput(
                summary="Demo scenario summary.",
                key_concerns=["[stub] Phase 13 demo data"],
                recommendation="manual_review",
                model_version="demo_fake_gemini_v1",
            )

    # Map each demo claim to the CV signature that matches its
    # construction in the seeders above.
    cv_signatures = {
        "CLM-DEMO-S1-LEGIT": ("scratch", "minor"),
        "CLM-DEMO-S2-INFLATED": ("dent", "minor"),
        "CLM-DEMO-S3-MISMATCH": ("dent", "minor"),
        "CLM-DEMO-S4-PREV": ("bumper_damage", "moderate"),
        "CLM-DEMO-S5-MULTI": ("dent", "minor"),
    }

    summaries: dict[int, dict] = {}
    for claim in claims:
        sig = cv_signatures.get(claim.claim_number, ("scratch", "minor"))
        predictor = _FakeCV(*sig)
        gemini = _FakeGemini()
        result = pipeline_service.run_analysis(
            claim.id, db,
            cv_predictor=predictor, gemini_client_obj=gemini,
        )
        db.refresh(claim)
        from app.models.risk_signal import RiskSignal
        rule_ids = sorted(s.rule_id for s in db.query(RiskSignal).filter(
            RiskSignal.claim_id == claim.id,
        ).all())
        summaries[claim.id] = {
            "claim_number": claim.claim_number,
            "status": result.status,
            "risk_band": claim.risk_band,
            "risk_score": claim.risk_score,
            "signal_count": result.signal_count,
            "signal_rule_ids": rule_ids,
        }
    return summaries


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--analyze", action="store_true",
        help="Run the analysis pipeline for every seeded claim after seeding.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete any existing demo data before seeding.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Reserved for future use; the data is hand-crafted so the seed "
             "has no effect today. Accepted for forward-compatibility.",
    )
    args = parser.parse_args()

    if args.seed != RANDOM_SEED:
        # The data is hand-crafted and deterministic; the seed is a
        # no-op today but we accept it so future random-data variants
        # of this script do not have to change the CLI.
        random.seed(args.seed)

    db = _ensure_session()
    try:
        if args.reset:
            print("Removing existing demo data…")
            _drop_demo_data(db)

        print("Seeding 5 demo scenarios (Section 3.3)…")
        claims: list[Claim] = []
        for sc in _SCENARIOS:
            claim = sc["seeder"](db)
            claims.append(claim)
            print(f"  {sc['key']} {sc['name']:<32}  claim #{claim.id}  "
                  f"({claim.claim_number})")
        db.commit()
        print(f"Seeded {len(claims)} claims.")

        if args.analyze:
            print("\nRunning the analysis pipeline for each claim…")
            summaries = _analyze_all_claims(db, claims)
            db.commit()
            print(f"\n{'Scenario':<8} {'Claim':<22} {'Band':<8} "
                  f"{'Score':<7} {'#sig':<5} Signals")
            print("-" * 110)
            for sc in _SCENARIOS:
                s = next(s for s in summaries.values()
                         if s["claim_number"] == sc["claim_number"])
                print(f"{sc['key']:<8} {s['claim_number']:<22} "
                      f"{(s['risk_band'] or '-'):<8} "
                      f"{(f'{s['risk_score']:.1f}' if s['risk_score'] is not None else '-'):<7} "
                      f"{s['signal_count']:<5} "
                      f"{','.join(s['signal_rule_ids']) or '∅'}")

        print("\nDone.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
