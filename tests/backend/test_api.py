"""
Phase 3 API integration tests.

Tests the FastAPI endpoints for customers, claims, documents, and images.
All tests use the `client` fixture for making requests and `db_session`
to seed or verify data. The `get_db` dependency is automatically overridden
by `db_session` in `conftest.py`.
"""

import pytest
import datetime
from httpx import Response
from io import BytesIO

from app.models.customer import Customer
from app.models.vehicle import Vehicle
from app.models.policy import Policy
from app.models.claim import Claim
from app.models.risk_signal import RiskSignal
from app.models.evidence import Evidence
from app.models.investigation import Investigation
from app.models.document import Document
from app.models.enums import (
    ClaimStatus,
    RiskBand,
    Recommendation,
    SignalSeverity,
    ExtractionStatus,
    DocType,
    EvidenceType,
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def seed_base_data(db_session):
    """Seed a customer, vehicle, and policy for claim creation."""
    c = Customer(name="John Doe", email="john@example.com", phone="1234567890")
    db_session.add(c)
    db_session.flush()
    
    v = Vehicle(customer_id=c.id, make="Toyota", model="Camry", year=2020, vin="VIN123", plate_number="ABC1234")
    db_session.add(v)
    db_session.flush()
    
    p = Policy(
        customer_id=c.id, vehicle_id=v.id, policy_number="POL123", coverage_type="comprehensive",
        coverage_limit=100000, deductible=500, start_date=datetime.date(2024, 1, 1), end_date=datetime.date(2025, 1, 1), status="active"
    )
    db_session.add(p)
    db_session.commit()
    
    return c, v, p

# ─── Customers API ────────────────────────────────────────────────────────────

def test_create_customer(client, db_session):
    response = client.post(
        "/customers",
        json={"name": "Alice API", "email": "alice_api@example.com", "phone": "555-1234"}
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Alice API"
    assert "id" in data

def test_create_customer_conflict(client, db_session):
    # First creation
    client.post("/customers", json={"name": "Bob", "email": "bob@example.com"})
    # Second creation with same email
    response = client.post("/customers", json={"name": "Bob2", "email": "bob@example.com"})
    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]

# ─── Claims API ───────────────────────────────────────────────────────────────

def test_create_claim(client, db_session):
    c, v, p = seed_base_data(db_session)
    
    response = client.post(
        "/claims",
        json={
            "claim_number": "CLM-API-001",
            "policy_id": p.id,
            "vehicle_id": v.id,
            "incident_date": "2024-06-01",
            "claimed_amount": 1500.00
        }
    )
    assert response.status_code == 201
    data = response.json()
    assert data["claim_number"] == "CLM-API-001"
    assert data["status"] == "pending"
    assert data["policy_id"] == p.id

def test_create_claim_invalid_policy(client, db_session):
    response = client.post(
        "/claims",
        json={
            "claim_number": "CLM-ERR-001",
            "policy_id": 9999,
            "vehicle_id": 9999,
            "incident_date": "2024-06-01"
        }
    )
    assert response.status_code == 404
    assert "Policy not found" in response.json()["detail"]

def test_list_claims(client, db_session):
    c, v, p = seed_base_data(db_session)
    client.post("/claims", json={"claim_number": "CLM-1", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    client.post("/claims", json={"claim_number": "CLM-2", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-02"})
    
    response = client.get("/claims")
    assert response.status_code == 200
    assert len(response.json()) == 2

def test_get_claim_detail(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-3", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    
    response = client.get(f"/claims/{claim_id}")
    assert response.status_code == 200
    assert response.json()["claim_number"] == "CLM-3"

def test_get_claim_detail_not_found(client, db_session):
    response = client.get("/claims/99999")
    assert response.status_code == 404

# ─── Documents API ────────────────────────────────────────────────────────────

def test_upload_document(client, db_session, monkeypatch):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DOC-1", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    
    # Mock storage to avoid actual disk writes during test
    monkeypatch.setattr("app.api.documents.storage.save_upload_file", lambda f, i: "uploads/test/doc.pdf")
    monkeypatch.setattr("app.api.documents.storage.delete_file", lambda p: None)
    
    response = client.post(
        f"/claims/{claim_id}/documents",
        data={"doc_type": "claim_form"},
        files={"file": ("test.pdf", b"dummy pdf content", "application/pdf")}
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["doc_type"] == "claim_form"
    assert data["file_path"] == "uploads/test/doc.pdf"
    assert data["extraction_status"] == "pending"

def test_upload_document_invalid_type(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DOC-2", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    
    response = client.post(
        f"/claims/{claim_id}/documents",
        data={"doc_type": "claim_form"},
        files={"file": ("test.txt", b"dummy txt content", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

# ─── Images API ───────────────────────────────────────────────────────────────

def test_upload_images(client, db_session, monkeypatch):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-IMG-1", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    
    monkeypatch.setattr("app.api.images.storage.save_upload_file", lambda f, i: "uploads/test/img.jpg")
    monkeypatch.setattr("app.api.images.storage.delete_file", lambda p: None)
    
    response = client.post(
        f"/claims/{claim_id}/images",
        files=[
            ("files", ("test1.jpg", b"dummy image 1", "image/jpeg")),
            ("files", ("test2.png", b"dummy image 2", "image/png"))
        ]
    )
    
    assert response.status_code == 201
    data = response.json()
    assert len(data) == 2
    assert data[0]["source"] == "image"
    assert data[1]["source"] == "image"

# ─── Decision API ─────────────────────────────────────────────────────────────

def test_record_decision(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DEC-1", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    
    response = client.post(
        f"/claims/{claim_id}/decision",
        json={"decision": "approve", "notes": "Looks good"}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "decided"

def test_record_decision_conflict(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DEC-2", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]

    # First decision
    client.post(f"/claims/{claim_id}/decision", json={"decision": "approve"})

    # Second decision fails
    response = client.post(f"/claims/{claim_id}/decision", json={"decision": "deny"})
    assert response.status_code == 409
    assert "already been decided" in response.json()["detail"]


# ─── Phase 9: GET /customers ──────────────────────────────────────────────────

def test_list_customers(client, db_session):
    seed_base_data(db_session)  # creates one customer
    response = client.get("/customers")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["email"] == "john@example.com"
    # Lightweight summary: id, name, email only
    assert set(data[0].keys()) == {"id", "name", "email"}


def test_list_customers_empty(client, db_session):
    response = client.get("/customers")
    assert response.status_code == 200
    assert response.json() == []


# ─── Phase 9: /policies and /vehicles CRUD ────────────────────────────────────

def test_create_vehicle(client, db_session):
    c, v, p = seed_base_data(db_session)
    response = client.post(
        "/vehicles",
        json={
            "customer_id": c.id,
            "make": "Honda",
            "model": "Civic",
            "year": 2019,
            "vin": "VIN2HONDACIVIC",
            "plate_number": "XYZ9876",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["make"] == "Honda"
    assert data["customer_id"] == c.id


def test_list_vehicles(client, db_session):
    c, v, p = seed_base_data(db_session)
    client.post(
        "/vehicles",
        json={"customer_id": c.id, "make": "Honda", "model": "Civic", "year": 2019},
    )
    response = client.get("/vehicles")
    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_create_policy(client, db_session):
    c, v, p = seed_base_data(db_session)
    response = client.post(
        "/policies",
        json={
            "customer_id": c.id,
            "vehicle_id": v.id,
            "policy_number": "POL-EXTRA-001",
            "coverage_type": "collision",
            "coverage_limit": 50000,
            "deductible": 250,
            "start_date": "2024-01-01",
            "end_date": "2025-01-01",
        },
    )
    assert response.status_code == 201
    assert response.json()["policy_number"] == "POL-EXTRA-001"


def test_list_policies(client, db_session):
    c, v, p = seed_base_data(db_session)
    response = client.get("/policies")
    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["policy_number"] == "POL123"


def test_list_policies_filtered_by_customer(client, db_session):
    c, v, p = seed_base_data(db_session)
    response = client.get(f"/policies?customer_id={c.id}")
    assert response.status_code == 200
    assert len(response.json()) == 1
    response = client.get("/policies?customer_id=9999")
    assert response.status_code == 200
    assert response.json() == []


# ─── Phase 9: documents list ─────────────────────────────────────────────────

def test_list_documents_empty(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DOC-LIST", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    response = client.get(f"/claims/{claim_id}/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_returns_uploaded(client, db_session, monkeypatch):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DOC-LIST-2", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]

    monkeypatch.setattr("app.api.documents.storage.save_upload_file", lambda f, i: "uploads/test/list-a.pdf")
    monkeypatch.setattr("app.api.documents.storage.delete_file", lambda p: None)
    client.post(
        f"/claims/{claim_id}/documents",
        data={"doc_type": "claim_form"},
        files={"file": ("a.pdf", b"pdf a", "application/pdf")},
    )
    monkeypatch.setattr("app.api.documents.storage.save_upload_file", lambda f, i: "uploads/test/list-b.pdf")
    client.post(
        f"/claims/{claim_id}/documents",
        data={"doc_type": "estimate"},
        files={"file": ("b.pdf", b"pdf b", "application/pdf")},
    )

    response = client.get(f"/claims/{claim_id}/documents")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    # The endpoint sorts by created_at desc. SQLite's `now()` has 1s
    # resolution so both rows can land in the same instant — assert on
    # the set of types rather than the strict order.
    assert {row["doc_type"] for row in data} == {"claim_form", "estimate"}
    # DocumentSummary fields
    assert set(data[0].keys()) == {"id", "doc_type", "extraction_status", "file_path"}


def test_list_documents_404_for_missing_claim(client, db_session):
    response = client.get("/claims/9999/documents")
    assert response.status_code == 404


def test_get_document_includes_extracted_fields(client, db_session, monkeypatch):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DOC-DETAIL", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]

    monkeypatch.setattr("app.api.documents.storage.save_upload_file", lambda f, i: "uploads/test/detail.pdf")
    monkeypatch.setattr("app.api.documents.storage.delete_file", lambda p: None)
    upload_resp = client.post(
        f"/claims/{claim_id}/documents",
        data={"doc_type": "claim_form"},
        files={"file": ("detail.pdf", b"pdf", "application/pdf")},
    )
    document_id = upload_resp.json()["id"]

    # Direct DB update so the test doesn't depend on the extraction pipeline.
    document = db_session.get(Document, document_id)
    document.extracted_fields = {"policy_number": "POL123", "plate_number": "ABC1234"}
    document.raw_confidence = 0.85
    db_session.commit()

    response = client.get(f"/claims/{claim_id}/documents/{document_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == document_id
    assert data["extracted_fields"]["policy_number"] == "POL123"
    assert data["raw_confidence"] == 0.85


def test_get_document_404_when_not_in_claim(client, db_session, monkeypatch):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DOC-OTHER", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]

    monkeypatch.setattr("app.api.documents.storage.save_upload_file", lambda f, i: "uploads/test/x.pdf")
    monkeypatch.setattr("app.api.documents.storage.delete_file", lambda p: None)
    upload_resp = client.post(
        f"/claims/{claim_id}/documents",
        data={"doc_type": "claim_form"},
        files={"file": ("x.pdf", b"pdf", "application/pdf")},
    )
    document_id = upload_resp.json()["id"]

    # Other claim id, same document id → 404
    response = client.get(f"/claims/9999/documents/{document_id}")
    assert response.status_code == 404


# ─── Phase 9: investigation key_concerns + disclaimer ───────────────────────

def _seed_investigation(
    db_session, claim_id: int, summary: str, recommendation: str,
    *, model_version: str | None = None,
) -> None:
    inv = Investigation(
        claim_id=claim_id,
        summary_text=summary,
        recommendation=recommendation,
        model_version=model_version,
    )
    db_session.add(inv)
    db_session.commit()


def test_investigation_202_when_no_summary(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-INV-1", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    _seed_investigation(db_session, claim_id, summary="", recommendation="manual_review")
    response = client.get(f"/claims/{claim_id}/investigation")
    assert response.status_code == 202


def test_investigation_404_when_no_row(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-INV-2", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    response = client.get(f"/claims/{claim_id}/investigation")
    assert response.status_code == 404


def test_investigation_returns_key_concerns_and_disclaimer(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-INV-3", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    _seed_investigation(db_session, claim_id, summary="High repair cost relative to baseline.", recommendation="investigate")

    # Seed two risk signals
    sig1 = RiskSignal(claim_id=claim_id, rule_id="R4_excessive_repair_cost", category="cost",
                      severity=SignalSeverity.high, description="Repair cost is 2.5x the segment baseline.")
    sig2 = RiskSignal(claim_id=claim_id, rule_id="R1_unsupported_damage", category="document",
                      severity=SignalSeverity.medium, description="Damage summary not supported by photos.")
    db_session.add_all([sig1, sig2])
    db_session.commit()

    response = client.get(f"/claims/{claim_id}/investigation")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "High repair cost relative to baseline."
    assert data["recommendation"] == "investigate"
    assert data["disclaimer"] == "AI-generated, human decision required"
    assert len(data["key_concerns"]) == 2
    # Concerns are derived from RiskSignal rows; rule_id prefix is preserved.
    assert any(concern.startswith("[R4_excessive_repair_cost]") for concern in data["key_concerns"])
    assert any(concern.startswith("[R1_unsupported_damage]") for concern in data["key_concerns"])


def test_investigation_response_includes_model_version(client, db_session):
    """The GET /claims/{id}/investigation response must surface
    `model_version` so the frontend can label demo-mode summaries
    honestly (Issue 6, final bug-fix pass). When the row has no
    model_version recorded, the field is `null`.
    """
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-INV-MV", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    _seed_investigation(
        db_session, claim_id,
        summary="Demo summary.",
        recommendation="normal",
        model_version="demo_deterministic_v1",
    )

    response = client.get(f"/claims/{claim_id}/investigation")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data, "model_version must be in the response schema"
    assert data["model_version"] == "demo_deterministic_v1"


def test_investigation_response_model_version_null_when_unset(client, db_session):
    """When the Investigation row has no model_version recorded, the
    field is exposed as `null` (not omitted) so the frontend can render
    a consistent UI.
    """
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-INV-MV-NULL", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]
    _seed_investigation(
        db_session, claim_id,
        summary="Legacy summary.",
        recommendation="normal",
    )

    response = client.get(f"/claims/{claim_id}/investigation")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert data["model_version"] is None


# ─── Phase 9: decision notes persistence ────────────────────────────────────

def test_record_decision_persists_notes(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DEC-NOTES", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]

    response = client.post(
        f"/claims/{claim_id}/decision",
        json={"decision": "manual_review", "notes": "Need second opinion on rear-bumper damage."},
    )
    assert response.status_code == 200
    # Notes are surfaced on subsequent GET
    get_resp = client.get(f"/claims/{claim_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["decision_notes"] == "Need second opinion on rear-bumper damage."


def test_record_decision_without_notes_is_null(client, db_session):
    c, v, p = seed_base_data(db_session)
    create_resp = client.post("/claims", json={"claim_number": "CLM-DEC-NONOTES", "policy_id": p.id, "vehicle_id": v.id, "incident_date": "2024-06-01"})
    claim_id = create_resp.json()["id"]

    response = client.post(
        f"/claims/{claim_id}/decision",
        json={"decision": "approve"},
    )
    assert response.status_code == 200
    get_resp = client.get(f"/claims/{claim_id}")
    assert get_resp.json()["decision_notes"] is None


# ─── Phase 9: decision_notes column exists on the model ─────────────────────

def test_decision_notes_column_exists():
    """The Claim model must expose a nullable decision_notes column.

    No Alembic is wired in this project; tables are created by
    Base.metadata.create_all, which is exercised by every test that
    uses the `db_session` fixture. This test pins the column as part of
    the model contract so a future refactor cannot silently drop it.
    """
    from sqlalchemy import inspect
    from app.models.claim import Claim
    mapper = inspect(Claim)
    columns = {c.key for c in mapper.columns}
    assert "decision_notes" in columns
    # Nullable (defaults to None for undecided claims)
    assert "decision_notes" in {c.key for c in mapper.columns if c.nullable}


# ─── Phase 10: evidence bundle per risk signal ───────────────────────────────


def _seed_signal_with_all_evidence_types(db_session, claim_id: int) -> int:
    """Seed one RiskSignal and one Evidence row of each type, return the signal id.

    The detail_json shapes mirror what the production consistency
    rules and CV pipeline are expected to write. They are tested
    here so the evidence bundle returned by GET /claims/{id}/evidence
    is contract-pinned for the Phase 10 Evidence UI.
    """
    signal = RiskSignal(
        claim_id=claim_id,
        rule_id="R10_evidence_bundle",
        category="phase10_test",
        severity=SignalSeverity.medium.value,
        description="Phase 10 evidence-bundle test signal.",
    )
    db_session.add(signal)
    db_session.flush()

    db_session.add_all(
        [
            Evidence(
                risk_signal_id=signal.id,
                evidence_type=EvidenceType.image.value,
                reference="1",
                detail_json={
                    "confidence": 0.87,
                    "bounding_box": [10, 20, 100, 120],
                },
            ),
            Evidence(
                risk_signal_id=signal.id,
                evidence_type=EvidenceType.document.value,
                reference="7",
                detail_json={
                    "page": 2,
                    "field_name": "policy_number",
                    "value": "POL-EXTRACTED-9",
                    "confidence": 0.92,
                },
            ),
            Evidence(
                risk_signal_id=signal.id,
                evidence_type=EvidenceType.field.value,
                reference="incident_date",
                detail_json={
                    "field_name": "incident_date",
                    "expected": "2024-06-01",
                    "actual": "2024-05-15",
                    "source_a": "claim_form",
                    "source_b": "police_report",
                },
            ),
            Evidence(
                risk_signal_id=signal.id,
                evidence_type=EvidenceType.computed.value,
                reference="cost_ratio",
                detail_json={
                    "baseline_range": [800, 1400],
                    "claimed": 3200,
                    "ratio": 2.29,
                },
            ),
        ]
    )
    db_session.commit()
    return signal.id


def test_evidence_endpoint_returns_all_four_types(client, db_session):
    """GET /claims/{id}/evidence returns each evidence row with the
    right type, reference, and detail_json fields the UI consumes."""
    c, v, p = seed_base_data(db_session)
    create_resp = client.post(
        "/claims",
        json={
            "claim_number": "CLM-EV-ALL",
            "policy_id": p.id,
            "vehicle_id": v.id,
            "incident_date": "2024-06-01",
        },
    )
    claim_id = create_resp.json()["id"]

    _seed_signal_with_all_evidence_types(db_session, claim_id)

    response = client.get(f"/claims/{claim_id}/evidence")
    assert response.status_code == 200
    signals = response.json()
    assert len(signals) == 1
    evidence = signals[0]["evidence"]
    assert len(evidence) == 4

    by_type = {e["evidence_type"]: e for e in evidence}

    # Image: bounding box + confidence
    img = by_type[EvidenceType.image.value]
    assert img["reference"] == "1"
    assert img["detail_json"]["confidence"] == 0.87
    assert img["detail_json"]["bounding_box"] == [10, 20, 100, 120]

    # Document: page, field, value, confidence
    doc = by_type[EvidenceType.document.value]
    assert doc["reference"] == "7"
    assert doc["detail_json"]["page"] == 2
    assert doc["detail_json"]["field_name"] == "policy_number"
    assert doc["detail_json"]["value"] == "POL-EXTRACTED-9"
    assert doc["detail_json"]["confidence"] == 0.92

    # Field: expected vs actual
    fld = by_type[EvidenceType.field.value]
    assert fld["reference"] == "incident_date"
    assert fld["detail_json"]["expected"] == "2024-06-01"
    assert fld["detail_json"]["actual"] == "2024-05-15"
    assert fld["detail_json"]["source_a"] == "claim_form"
    assert fld["detail_json"]["source_b"] == "police_report"

    # Computed: baseline + claimed + ratio
    comp = by_type[EvidenceType.computed.value]
    assert comp["reference"] == "cost_ratio"
    assert comp["detail_json"]["baseline_range"] == [800, 1400]
    assert comp["detail_json"]["claimed"] == 3200
    assert comp["detail_json"]["ratio"] == 2.29


def test_evidence_endpoint_handles_missing_detail_json(client, db_session):
    """Evidence rows with no detail_json still surface in the response
    (the UI must render a graceful "not available" state for these)."""
    c, v, p = seed_base_data(db_session)
    create_resp = client.post(
        "/claims",
        json={
            "claim_number": "CLM-EV-MIN",
            "policy_id": p.id,
            "vehicle_id": v.id,
            "incident_date": "2024-06-01",
        },
    )
    claim_id = create_resp.json()["id"]

    signal = RiskSignal(
        claim_id=claim_id,
        rule_id="R10_min",
        category="phase10_test",
        severity=SignalSeverity.low.value,
        description="Minimal evidence",
    )
    db_session.add(signal)
    db_session.flush()
    db_session.add(
        Evidence(
            risk_signal_id=signal.id,
            evidence_type=EvidenceType.computed.value,
            reference=None,
            detail_json=None,
        )
    )
    db_session.commit()

    response = client.get(f"/claims/{claim_id}/evidence")
    assert response.status_code == 200
    signals = response.json()
    assert len(signals) == 1
    assert len(signals[0]["evidence"]) == 1
    assert signals[0]["evidence"][0]["detail_json"] is None
    assert signals[0]["evidence"][0]["reference"] is None


def test_evidence_endpoint_image_without_bounding_box(client, db_session):
    """Image evidence with a confidence but no bounding_box is a
    legal shape — the UI must not invent a box. We pin the contract
    that the detail_json is passed through verbatim."""
    c, v, p = seed_base_data(db_session)
    create_resp = client.post(
        "/claims",
        json={
            "claim_number": "CLM-EV-IMG-NOBBOX",
            "policy_id": p.id,
            "vehicle_id": v.id,
            "incident_date": "2024-06-01",
        },
    )
    claim_id = create_resp.json()["id"]

    signal = RiskSignal(
        claim_id=claim_id,
        rule_id="R10_img_nobox",
        category="phase10_test",
        severity=SignalSeverity.medium.value,
        description="Image without bbox",
    )
    db_session.add(signal)
    db_session.flush()
    db_session.add(
        Evidence(
            risk_signal_id=signal.id,
            evidence_type=EvidenceType.image.value,
            reference="2",
            detail_json={"confidence": 0.55},
        )
    )
    db_session.commit()

    response = client.get(f"/claims/{claim_id}/evidence")
    assert response.status_code == 200
    ev = response.json()[0]["evidence"][0]
    assert ev["evidence_type"] == "image"
    # No bounding_box key in detail_json — the UI shows "—" for it.
    assert "bounding_box" not in (ev["detail_json"] or {})
    assert ev["detail_json"]["confidence"] == 0.55


def test_evidence_endpoint_404_for_missing_claim(client, db_session):
    response = client.get("/claims/9999/evidence")
    assert response.status_code == 404
