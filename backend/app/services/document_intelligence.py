"""Document Intelligence — Phase 11 deterministic stub.

Blueprint Section 4 calls for an OCR / DocIntel provider that turns
uploaded claim documents into structured `extracted_fields`. Phase 11
ships a minimal, deterministic stub so the rest of the pipeline
(consistency R9, evidence generation, frontend Document Viewer) can
exercise the full data path. The real provider is a Phase 13 task.

What this stub does:
- For a Document with `extraction_status == "pending"`, look at the
  file on disk + the `doc_type`, and write a small JSON payload to
  `extracted_fields`.
- Always sets `raw_confidence` to 0.5 (calibrated, not overclaimed).
- Flips `extraction_status` to `"completed"` on success, `"failed"`
  on error (file missing, decode failure).

What it deliberately does NOT do:
- Call any external API or OCR service.
- Invent values it cannot derive from the file name. The "policy"
  branch is the only one that extracts anything, and only when the
  filename contains a token that looks like a policy number. We never
  claim to have read the file body.

The function is safe to call repeatedly on the same Document; the
caller is expected to skip rows whose `extraction_status != "pending"`.

Phase 14 (final bug-fix pass) — internal marker payload
{"_phase11_stub": True, "doc_type": ...} was previously written when
no filename-token could be parsed. That marker is no longer emitted:
the UI was surfacing the raw key/value pair to the end user, which
read as an internal implementation detail. The stub now returns
honest empty field sets derived from the file's metadata (filename +
doc_type) so the UI can show a meaningful "no structured fields
extracted" state instead of a `_phase11_stub` flag. The presence of
the stub itself is documented at the API/UI level (e.g. via the
"AI-generated" disclaimer on the investigation page).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document
from app.models.enums import ExtractionStatus

logger = logging.getLogger(__name__)


# Filename token that looks like a policy number: e.g. "POL-12345", "pol_12345".
_POLICY_NUMBER_RE = re.compile(r"(?i)\bpol[_-]?([a-z0-9]{4,})\b")

# Default confidence for the stub. Documented as 0.5 in the writeup
# so the UI can render it as "low confidence" — we never claim more
# than we can back up.
STUB_CONFIDENCE = 0.5


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
        fields = _run_stub_extraction(document)
    except Exception as exc:  # noqa: BLE001
        logger.exception("extract_document failed for doc %s: %s", document_id, exc)
        _mark_failed(document, db, reason=str(exc)[:500])
        return False

    document.extracted_fields = fields
    document.raw_confidence = STUB_CONFIDENCE
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


def _filename_of(document: Document) -> str:
    return Path(document.file_path).name


def _run_stub_extraction(document: Document) -> dict[str, Any]:
    """Derive a small JSON payload from the file's name and doc_type.

    We never read the file body in Phase 11. The only field we attempt
    to extract is `policy_number` from a "policy" document whose
    filename contains a token that looks like a policy number. For
    every other case we return an *empty* payload — there are no
    structured fields to surface, so we say so honestly.

    Earlier revisions wrote a `{"_phase11_stub": True, "doc_type": ...}`
    marker payload, but that surfaced internal implementation details
    to the end user. The UI now renders an empty fields object as
    "no structured fields extracted" instead.
    """
    filename = _filename_of(document)
    doc_type = document.doc_type

    # 1. If the file is missing on disk, we treat that as a real
    #    failure (not a "no fields" result) so the pipeline flips
    #    extraction_status to "failed". This surfaces uploads that
    #    were recorded but never made it to disk.
    full_path = _resolve_path(document.file_path)
    if not full_path.exists():
        raise FileNotFoundError(f"document file missing: {document.file_path}")

    if doc_type == "policy":
        match = _POLICY_NUMBER_RE.search(filename)
        if match:
            return {"policy_number": match.group(0).upper()}
        return {}

    # claim_form / estimate / invoice / previous_claim — we never
    # invent values; the file body is not read in the Phase 11 stub.
    return {}


# Re-export so test code can import the regex if it needs to assert.
__all__ = [
    "extract_document",
    "STUB_CONFIDENCE",
]
