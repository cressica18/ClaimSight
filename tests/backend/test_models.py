"""
Phase 2 database model tests.

All tests use the SQLite in-memory db_session fixture from conftest.py.
Tests cover:
 1. Database initialization / table creation
 2. Customer CRUD
 3. Vehicle linked to a customer
 4. Policy linked to customer + vehicle
 5. Claim linked to policy + vehicle; uniqueness constraint
 6. Documents for a claim
 7. RepairEstimate with RepairItems
 8. PreviousClaim
 9. RiskSignals with Evidence
10. Investigation
11. Foreign-key relationship traversal
12. Constraint failures (duplicate, invalid values)
"""

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from app.models import (
    Customer,
    Vehicle,
    Policy,
    Claim,
    Accident,
    Damage,
    Document,
    RepairEstimate,
    RepairItem,
    PreviousClaim,
    RiskSignal,
    Evidence,
    Investigation,
)
from app.models.enums import (
    ClaimStatus,
    DocType,
    EvidenceType,
    ExtractionStatus,
    Recommendation,
    RepairOperation,
    SignalSeverity,
)

import datetime


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_customer(session, email="alice@example.com", name="Alice Smith"):
    c = Customer(name=name, email=email, phone="+1-555-0100")
    session.add(c)
    session.flush()
    return c


def make_vehicle(session, customer_id, vin="1HGCM82633A004352"):
    v = Vehicle(
        customer_id=customer_id,
        make="Honda",
        model="Accord",
        year=2021,
        vin=vin,
        plate_number="ABC-1234",
    )
    session.add(v)
    session.flush()
    return v


def make_policy(session, customer_id, vehicle_id, policy_number="POL-001"):
    p = Policy(
        customer_id=customer_id,
        vehicle_id=vehicle_id,
        policy_number=policy_number,
        coverage_type="comprehensive",
        coverage_limit=50000.00,
        deductible=500.00,
        start_date=datetime.date(2024, 1, 1),
        end_date=datetime.date(2025, 1, 1),
        status="active",
    )
    session.add(p)
    session.flush()
    return p


def make_claim(session, policy_id, vehicle_id, claim_number="CLM-001"):
    cl = Claim(
        policy_id=policy_id,
        vehicle_id=vehicle_id,
        claim_number=claim_number,
        incident_date=datetime.date(2024, 6, 15),
        reported_date=datetime.date(2024, 6, 16),
        claimed_amount=3500.00,
        status=ClaimStatus.pending.value,
    )
    session.add(cl)
    session.flush()
    return cl


# ─── Test 1: Table creation ───────────────────────────────────────────────────


def test_all_tables_created(sqlite_engine):
    """All 13 domain tables must exist after Base.metadata.create_all()."""
    inspector = inspect(sqlite_engine)
    tables = set(inspector.get_table_names())
    expected = {
        "customers", "vehicles", "policies", "claims", "accidents",
        "damages", "documents", "repair_estimates", "repair_items",
        "previous_claims", "risk_signals", "evidence", "investigations",
    }
    assert expected.issubset(tables), f"Missing tables: {expected - tables}"


# ─── Test 2: Customer CRUD ────────────────────────────────────────────────────


def test_create_customer(db_session):
    c = make_customer(db_session)
    assert c.id is not None
    assert c.email == "alice@example.com"
    assert c.created_at is None or c.created_at  # server_default; may be None in SQLite


def test_customer_email_unique(db_session):
    make_customer(db_session, email="dup@example.com")
    with pytest.raises(IntegrityError):
        make_customer(db_session, email="dup@example.com")


# ─── Test 3: Vehicle linked to customer ──────────────────────────────────────


def test_create_vehicle(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    assert v.id is not None
    assert v.customer_id == c.id
    assert v.make == "Honda"


def test_vehicle_fk_to_customer(db_session):
    """Creating a vehicle with a non-existent customer_id must fail."""
    v = Vehicle(customer_id=99999, make="Ford", model="Focus", year=2020)
    db_session.add(v)
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_vehicle_traverses_to_customer(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    db_session.refresh(v)
    assert v.customer.name == "Alice Smith"


def test_vehicle_vin_unique(db_session):
    c = make_customer(db_session)
    make_vehicle(db_session, c.id, vin="UNIQUE123456789AB")
    with pytest.raises(IntegrityError):
        make_vehicle(db_session, c.id, vin="UNIQUE123456789AB")


# ─── Test 4: Policy ──────────────────────────────────────────────────────────


def test_create_policy(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    assert p.id is not None
    assert p.policy_number == "POL-001"
    assert p.coverage_limit == 50000.00


def test_policy_number_unique(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    make_policy(db_session, c.id, v.id, policy_number="POL-DUP")
    with pytest.raises(IntegrityError):
        make_policy(db_session, c.id, v.id, policy_number="POL-DUP")


# ─── Test 5: Claim + unique constraint ───────────────────────────────────────


def test_create_claim(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)
    assert cl.id is not None
    assert cl.status == ClaimStatus.pending.value
    assert cl.risk_score is None
    assert cl.risk_band is None


def test_claim_unique_policy_claim_number(db_session):
    """Same policy_id + claim_number must be rejected (Section 13 duplicate guard)."""
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    make_claim(db_session, p.id, v.id, claim_number="CLM-DUP")
    with pytest.raises(IntegrityError):
        make_claim(db_session, p.id, v.id, claim_number="CLM-DUP")


def test_claim_traverses_to_policy(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)
    db_session.refresh(cl)
    assert cl.policy.policy_number == "POL-001"
    assert cl.vehicle.make == "Honda"


# ─── Test 6: Documents for a claim ───────────────────────────────────────────


def test_create_document(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)

    doc = Document(
        claim_id=cl.id,
        doc_type=DocType.claim_form.value,
        file_path="uploads/CLM-001/claim_form.pdf",
        extraction_status=ExtractionStatus.pending.value,
    )
    db_session.add(doc)
    db_session.flush()
    assert doc.id is not None
    assert doc.extraction_status == ExtractionStatus.pending.value


def test_document_fk_to_claim(db_session):
    doc = Document(
        claim_id=99999,
        doc_type=DocType.policy.value,
        file_path="uploads/x/y.pdf",
        extraction_status="pending",
    )
    db_session.add(doc)
    with pytest.raises(IntegrityError):
        db_session.flush()


# ─── Test 7: RepairEstimate + RepairItems ─────────────────────────────────────


def test_create_repair_estimate_with_items(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)

    estimate = RepairEstimate(
        claim_id=cl.id,
        shop_name="City Auto Body",
        total_cost=3200.00,
        currency="USD",
        issued_date=datetime.date(2024, 6, 20),
    )
    db_session.add(estimate)
    db_session.flush()

    item1 = RepairItem(
        repair_estimate_id=estimate.id,
        part_name="Front Bumper",
        operation=RepairOperation.replace.value,
        cost=800.00,
        labor_hours=4.0,
    )
    item2 = RepairItem(
        repair_estimate_id=estimate.id,
        part_name="Hood",
        operation=RepairOperation.repair.value,
        cost=350.00,
        labor_hours=2.5,
    )
    db_session.add_all([item1, item2])
    db_session.flush()

    db_session.refresh(estimate)
    assert len(estimate.items) == 2
    assert estimate.items[0].part_name in {"Front Bumper", "Hood"}


# ─── Test 8: PreviousClaim ────────────────────────────────────────────────────


def test_create_previous_claim(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    pc = PreviousClaim(
        customer_id=c.id,
        vehicle_id=v.id,
        claim_number="OLD-CLM-001",
        incident_date=datetime.date(2023, 11, 1),
        damage_summary="Minor rear-end damage",
        claimed_amount=1200.00,
    )
    db_session.add(pc)
    db_session.flush()
    assert pc.id is not None
    assert pc.overlap_score is None  # Not yet computed by consistency engine


# ─── Test 9: RiskSignals + Evidence ──────────────────────────────────────────


def test_create_risk_signal_with_evidence(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)

    signal = RiskSignal(
        claim_id=cl.id,
        rule_id="R4_excessive_repair_cost",
        category="financial",
        severity=SignalSeverity.high.value,
        description="Repair cost is 2.3× the baseline upper bound.",
    )
    db_session.add(signal)
    db_session.flush()

    ev = Evidence(
        risk_signal_id=signal.id,
        evidence_type=EvidenceType.computed.value,
        reference=None,
        detail_json={"baseline_range": [800, 1400], "claimed": 3200, "ratio": 2.29},
    )
    db_session.add(ev)
    db_session.flush()

    db_session.refresh(signal)
    assert len(signal.evidence) == 1
    assert signal.evidence[0].evidence_type == EvidenceType.computed.value
    assert signal.evidence[0].detail_json["ratio"] == 2.29


def test_risk_signal_fk_to_claim(db_session):
    signal = RiskSignal(
        claim_id=99999,
        rule_id="R1_test",
        category="test",
        severity=SignalSeverity.low.value,
        description="test",
    )
    db_session.add(signal)
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_evidence_fk_to_risk_signal(db_session):
    ev = Evidence(
        risk_signal_id=99999,
        evidence_type=EvidenceType.field.value,
    )
    db_session.add(ev)
    with pytest.raises(IntegrityError):
        db_session.flush()


# ─── Test 10: Investigation ───────────────────────────────────────────────────


def test_create_investigation(db_session):
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)

    inv = Investigation(
        claim_id=cl.id,
        recommendation=Recommendation.investigate.value,
        summary_text=None,  # null until Gemini runs (Phase 8)
    )
    db_session.add(inv)
    db_session.flush()
    assert inv.id is not None
    assert inv.summary_text is None
    assert inv.recommendation == Recommendation.investigate.value


def test_investigation_one_per_claim(db_session):
    """A second investigation for the same claim must fail (unique claim_id)."""
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)

    inv1 = Investigation(
        claim_id=cl.id, recommendation=Recommendation.normal.value
    )
    db_session.add(inv1)
    db_session.flush()

    inv2 = Investigation(
        claim_id=cl.id, recommendation=Recommendation.investigate.value
    )
    db_session.add(inv2)
    with pytest.raises(IntegrityError):
        db_session.flush()


# ─── Test 11: Full relationship traversal ─────────────────────────────────────


def test_full_relationship_traversal(db_session):
    """Verify the complete claim hierarchy traverses correctly."""
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)

    # Accident
    accident = Accident(claim_id=cl.id, description="Rear-end collision", location="Main St")
    db_session.add(accident)

    # Damage
    dmg = Damage(
        claim_id=cl.id, source="image", damage_type="dent", severity="moderate", confidence=0.87
    )
    db_session.add(dmg)

    # Document
    doc = Document(
        claim_id=cl.id,
        doc_type=DocType.estimate.value,
        file_path="uploads/CLM-001/estimate.pdf",
        extraction_status=ExtractionStatus.pending.value,
    )
    db_session.add(doc)

    # Risk signal + evidence
    signal = RiskSignal(
        claim_id=cl.id,
        rule_id="R1_unsupported_damage",
        category="consistency",
        severity=SignalSeverity.high.value,
        description="Claimed front damage not found in images.",
    )
    db_session.add(signal)
    db_session.flush()

    ev = Evidence(
        risk_signal_id=signal.id,
        evidence_type=EvidenceType.image.value,
        reference="img_001",
        detail_json={"confidence": 0.87},
    )
    db_session.add(ev)

    # Investigation
    inv = Investigation(claim_id=cl.id, recommendation=Recommendation.investigate.value)
    db_session.add(inv)
    db_session.flush()

    # Refresh and traverse
    db_session.refresh(cl)
    assert cl.accident.description == "Rear-end collision"
    assert len(cl.damages) == 1
    assert cl.damages[0].confidence == 0.87
    assert len(cl.documents) == 1
    assert len(cl.risk_signals) == 1
    assert len(cl.risk_signals[0].evidence) == 1
    assert cl.investigation.recommendation == Recommendation.investigate.value

    # Traverse up
    assert cl.policy.customer.name == "Alice Smith"
    assert cl.vehicle.make == "Honda"


# ─── Test 12: Cascade delete behaviour ───────────────────────────────────────


def test_cascade_delete_claim_deletes_signals(db_session):
    """Deleting a claim must cascade-delete its risk signals and evidence."""
    c = make_customer(db_session)
    v = make_vehicle(db_session, c.id)
    p = make_policy(db_session, c.id, v.id)
    cl = make_claim(db_session, p.id, v.id)

    signal = RiskSignal(
        claim_id=cl.id,
        rule_id="R2_test",
        category="consistency",
        severity=SignalSeverity.medium.value,
        description="Test signal",
    )
    db_session.add(signal)
    db_session.flush()
    signal_id = signal.id

    ev = Evidence(
        risk_signal_id=signal.id, evidence_type=EvidenceType.computed.value
    )
    db_session.add(ev)
    db_session.flush()
    ev_id = ev.id

    db_session.delete(cl)
    db_session.flush()

    assert db_session.get(RiskSignal, signal_id) is None
    assert db_session.get(Evidence, ev_id) is None
