"""Document Intelligence — real PDF text extraction with OCR fallback.

This module provides the public `extract_document` function used by the
analysis pipeline. It delegates to `DocumentExtractor` which performs
deterministic text extraction from PDF files and structures the results
into fields consumable by the consistency rules (especially R9).

Supported document types:
- policy: policy_number, coverage_type, policy_start_date, policy_end_date, insured_name, vin, plate_number
- claim_form: claim_number, policy_number, accident_date, claimed_amount, accident_description, claimed_damage_types, vin, plate_number
- estimate/invoice: total_estimate, repair_date, currency, repair_items, shop_name, policy_number, claim_number, vin, plate_number
- previous_claim: previous_claim_number, incident_date, damage_summary, claimed_amount, policy_number, vin, plate_number

The extractor first tries native PDF text extraction via PyMuPDF.
If the PDF appears to be scanned/image-only (minimal extractable text),
it falls back to OCR using Tesseract (if available). The extraction
method ("text", "ocr", "filename", "none") is stored in the extracted
fields for provenance. If OCR is unavailable and no text can be
extracted, the module returns an honest empty field set with a low
confidence score.

The function is safe to call repeatedly on the same Document; the
caller is expected to skip rows whose `extraction_status != "pending"`.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.enums import ExtractionStatus
from app.services.document_extractor import DocumentExtractor, ExtractionResult

logger = logging.getLogger(__name__)


def extract_document(db: Session, claim_id: int, document_id: int) -> bool:
    """Run document intelligence on a single document.

    Returns True on success (status flipped to "completed" with
    `extracted_fields` populated), False on failure (status flipped
    to "failed", `extracted_fields` left null). The function never
    raises out — callers can treat the return value as the source of
    truth and log accordingly.

    Idempotent: if the document's status is not "pending", returns
    True without changes (so re-running the pipeline is safe).
    """
    document = db.get(Document, document_id)
    if document is None or document.claim_id != claim_id:
        logger.warning(
            "extract_document: document %s not found for claim %s",
            document_id,
            claim_id,
        )
        return False

    if document.extraction_status != ExtractionStatus.pending.value:
        return True

    try:
        result = _run_extraction(document)
    except Exception as exc:  # noqa: BLE001
        logger.exception("extract_document failed for doc %s: %s", document_id, exc)
        _mark_failed(document, db, reason=str(exc)[:500])
        return False

    # Store user-facing fields plus extraction provenance in extracted_fields.
    # The extraction_method ("text", "ocr", "filename", "none") allows the UI
    # to indicate how the fields were derived. This is not an internal marker
    # but a user-facing provenance signal.
    fields = dict(result.fields)
    fields["extraction_method"] = result.extraction_method

    document.extracted_fields = fields
    document.raw_confidence = result.raw_confidence
    document.extraction_status = ExtractionStatus.completed.value
    db.add(document)
    db.commit()
    db.refresh(document)
    return True


# ─── Internals ──────────────────────────────────────────────────────────────


def _mark_failed(document: Document, db: Session, *, reason: str) -> None:
    """Flip the document to failed state without raising."""
    document.extraction_status = ExtractionStatus.failed.value
    document.raw_confidence = None
    # We deliberately do NOT write `extracted_fields` so a later
    # re-run can populate it cleanly.
    db.add(document)
    try:
        db.commit()
    except Exception:  # noqa: BLE001
        db.rollback()
        logger.exception("Failed to mark document %s as failed", document.id)


def _resolve_path(relative_path: str) -> Path:
    """Map a Document.file_path like "uploads/{claim_id}/{filename}"
    to an absolute Path on disk. Returns a non-existent Path if the
    file is missing — callers handle that as a soft failure.
    """
    base_dir = Path(settings.upload_dir)
    if not base_dir.is_absolute():
        base_dir = Path(os.getcwd()) / base_dir
    # relative_path looks like "uploads/{claim_id}/{filename}".
    parts = relative_path.split("/")
    if len(parts) >= 3 and parts[0] == "uploads":
        return base_dir / parts[1] / parts[2]
    return base_dir / relative_path


def _run_extraction(document: Document) -> ExtractionResult:
    """Extract structured fields from the document file using PyMuPDF."""
    full_path = _resolve_path(document.file_path)
    if not full_path.exists():
        raise FileNotFoundError(f"document file missing: {document.file_path}")

    # Only process PDF files; other types return empty with warning
    if full_path.suffix.lower() != ".pdf":
        logger.warning("Unsupported file type for extraction: %s", full_path.suffix)
        return ExtractionResult(
            fields={},
            raw_confidence=0.0,
            warnings=(f"Unsupported file type: {full_path.suffix}. Only PDF is supported.",),
            is_scanned=False,
        )

    extractor = DocumentExtractor()
    return extractor.extract(full_path, document.doc_type)


# Re-export for backward compatibility with tests
from app.services.document_extractor import STUB_CONFIDENCE  # noqa: E402

__all__ = [
    "extract_document",
    "STUB_CONFIDENCE",
]