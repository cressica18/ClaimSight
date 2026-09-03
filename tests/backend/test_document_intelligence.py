"""Tests for the Phase 11 document-intelligence stub.

The stub extracts structured fields from uploaded documents. In Phase 11
it deliberately does NOT read the file body — the only deterministic
extraction it can do is parse a `policy_number`-shaped token out of the
filename of a `policy` document. For every other case (or when a policy
filename has no such token) the stub must return an *empty* payload, so
the frontend can render a "no structured fields extracted" state instead
of an internal implementation marker.

Regression test for the Phase 14 (final bug-fix pass) defect: the stub
previously returned `{"_phase11_stub": True, "doc_type": "claim_form"}`
which the Document Viewer was rendering verbatim. The frontend was
exposing the `_phase11_stub` key to the end user. The stub now returns
an empty dict for the no-extractable-fields branch, and a focused
`{policy_number, _source}` payload only for the policy-with-token case.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from app.models.claim import Claim
from app.models.customer import Customer
from app.models.document import Document
from app.models.enums import ClaimStatus, DocType, ExtractionStatus
from app.models.policy import Policy
from app.models.vehicle import Vehicle
from app.services import document_intelligence


def _make_claim(db: Session, claim_number: str) -> int:
    """Create a Customer + Vehicle + Policy + Claim, return claim_id."""
    cust = Customer(
        name="DocStub Test",
        email=f"docstub-{claim_number.lower()}@test.com",
        phone="555-0001",
    )
    db.add(cust)
    db.flush()
    veh = Vehicle(
        customer_id=cust.id, make="Honda", model="Civic",
        year=2020, vin=f"DOCSTUBVIN-{claim_number}",
    )
    db.add(veh)
    db.flush()
    pol = Policy(
        customer_id=cust.id, vehicle_id=veh.id,
        policy_number=f"POL-{claim_number}", coverage_type="comprehensive",
        coverage_limit=50000.0, deductible=500.0,
        start_date=dt.date(2024, 1, 1), end_date=dt.date(2026, 12, 31),
        status="active",
    )
    db.add(pol)
    db.flush()
    claim = Claim(
        claim_number=claim_number,
        policy_id=pol.id,
        vehicle_id=veh.id,
        incident_date=dt.date(2026, 1, 15),
        reported_date=dt.date(2026, 1, 16),
        status=ClaimStatus.pending.value,
    )
    db.add(claim)
    db.flush()
    return claim.id


def _add_doc(
    db: Session,
    *,
    claim_id: int,
    doc_type: str,
    file_path: str,
) -> Document:
    doc = Document(
        claim_id=claim_id,
        doc_type=doc_type,
        file_path=file_path,
        extraction_status=ExtractionStatus.pending.value,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


@pytest.fixture
def patched_upload_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Make the document_intelligence stub resolve file paths to
    `tmp_path`. Each test writes the files it wants to be present.
    """
    from app.core.config import settings as _settings
    monkeypatch.setattr(_settings, "upload_dir", str(tmp_path))
    return tmp_path


def test_stub_does_not_emit_phase11_stub_marker_for_claim_form(
    db_session: Session, patched_upload_dir: Path
) -> None:
    """A `claim_form` document must not produce a payload that
    contains a `_phase11_stub` key. Earlier revisions wrote
    `{"_phase11_stub": True, "doc_type": "claim_form"}` and the UI
    rendered that key directly to the user.
    """
    claim_id = _make_claim(db_session, "DOCSTUB-CLAIM-FORM")
    (patched_upload_dir / str(claim_id)).mkdir(parents=True, exist_ok=True)
    (patched_upload_dir / str(claim_id) / "claim-form.pdf").write_bytes(b"%PDF-stub")
    doc = _add_doc(
        db_session,
        claim_id=claim_id,
        doc_type=DocType.claim_form.value,
        file_path=f"uploads/{claim_id}/claim-form.pdf",
    )
    document_intelligence.extract_document(db_session, doc.claim_id, doc.id)
    db_session.refresh(doc)

    assert doc.extraction_status == ExtractionStatus.completed.value
    assert doc.extracted_fields is not None
    assert "_phase11_stub" not in doc.extracted_fields, (
        "stub must not surface internal markers to the UI"
    )
    assert doc.extracted_fields == {}


def test_stub_does_not_emit_phase11_stub_marker_for_estimate(
    db_session: Session, patched_upload_dir: Path
) -> None:
    """An `estimate` document with no parseable token in the filename
    must also produce an empty payload, not a stub marker.
    """
    claim_id = _make_claim(db_session, "DOCSTUB-ESTIMATE")
    (patched_upload_dir / str(claim_id)).mkdir(parents=True, exist_ok=True)
    (patched_upload_dir / str(claim_id) / "repair-estimate.pdf").write_bytes(b"%PDF-stub")
    doc = _add_doc(
        db_session,
        claim_id=claim_id,
        doc_type=DocType.estimate.value,
        file_path=f"uploads/{claim_id}/repair-estimate.pdf",
    )
    document_intelligence.extract_document(db_session, doc.claim_id, doc.id)
    db_session.refresh(doc)

    assert doc.extraction_status == ExtractionStatus.completed.value
    assert doc.extracted_fields == {}


def test_stub_extracts_policy_number_when_filename_matches(
    db_session: Session, patched_upload_dir: Path
) -> None:
    """A `policy` document whose filename contains a `POL-XXXX` token
    must produce `{policy_number: "POL-99999"}` and nothing else —
    no internal `_source` marker, no `_phase11_stub` flag. The
    only user-facing field is `policy_number`.
    """
    claim_id = _make_claim(db_session, "DOCSTUB-POLICY")
    (patched_upload_dir / str(claim_id)).mkdir(parents=True, exist_ok=True)
    (patched_upload_dir / str(claim_id) / "POL-99999-policy.pdf").write_bytes(b"%PDF-stub")
    doc = _add_doc(
        db_session,
        claim_id=claim_id,
        doc_type=DocType.policy.value,
        file_path=f"uploads/{claim_id}/POL-99999-policy.pdf",
    )
    document_intelligence.extract_document(db_session, doc.claim_id, doc.id)
    db_session.refresh(doc)

    assert doc.extraction_status == ExtractionStatus.completed.value
    assert doc.extracted_fields == {"policy_number": "POL-99999"}
    # Defense against re-introducing internal markers.
    assert "_source" not in doc.extracted_fields
    assert "_phase11_stub" not in doc.extracted_fields


def test_stub_marks_missing_file_as_failed(
    db_session: Session, patched_upload_dir: Path
) -> None:
    """If the file is recorded in the DB but missing on disk, the
    stub must flip the document to `failed` — this surfaces uploads
    that were recorded but never made it to disk.
    """
    claim_id = _make_claim(db_session, "DOCSTUB-MISSING")
    doc = _add_doc(
        db_session,
        claim_id=claim_id,
        doc_type=DocType.claim_form.value,
        file_path=f"uploads/{claim_id}/does-not-exist.pdf",
    )
    document_intelligence.extract_document(db_session, doc.claim_id, doc.id)
    db_session.refresh(doc)

    assert doc.extraction_status == ExtractionStatus.failed.value
    assert doc.extracted_fields is None


def test_policy_payload_has_no_leading_underscore_keys(
    db_session: Session, patched_upload_dir: Path
) -> None:
    """The policy branch must not leak any keys beginning with `_`.

    This is a defense-in-depth test: even if a future revision
    re-introduces a `_source`/`_phase11_stub` debug marker, the
    test catches it at the data boundary before it can reach the
    API. The Document Viewer also filters leading-underscore keys
    at the UI boundary, so a leak in one place is caught by the
    other.
    """
    claim_id = _make_claim(db_session, "DOCSTUB-POLICY-LEAK")
    (patched_upload_dir / str(claim_id)).mkdir(parents=True, exist_ok=True)
    (patched_upload_dir / str(claim_id) / "POL-12345-coverage.pdf").write_bytes(b"%PDF-stub")
    doc = _add_doc(
        db_session,
        claim_id=claim_id,
        doc_type=DocType.policy.value,
        file_path=f"uploads/{claim_id}/POL-12345-coverage.pdf",
    )
    document_intelligence.extract_document(db_session, doc.claim_id, doc.id)
    db_session.refresh(doc)

    # Every key in the persisted payload must be user-facing.
    for key in doc.extracted_fields:
        assert not key.startswith("_"), (
            f"internal marker {key!r} leaked into the user-facing payload"
        )
