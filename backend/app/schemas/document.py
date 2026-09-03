"""Document Pydantic schemas."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.models.enums import DocType, ExtractionStatus


class DocumentBase(BaseModel):
    doc_type: DocType
    file_path: str = Field(..., min_length=1, max_length=1000)


class DocumentCreate(DocumentBase):
    """Created internally when a file is uploaded (Phase 3)."""
    claim_id: int


class Document(DocumentBase):
    """Full document response."""
    id: int
    claim_id: int
    extraction_status: ExtractionStatus
    raw_confidence: float | None
    # Phase 9: extracted fields surface in the Document Viewer side panel.
    # Nullable for backwards-compat with documents uploaded before the
    # extraction pipeline ran.
    extracted_fields: dict[str, Any] | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentSummary(BaseModel):
    """Lightweight document view."""
    id: int
    doc_type: DocType
    extraction_status: ExtractionStatus
    file_path: str

    model_config = {"from_attributes": True}
