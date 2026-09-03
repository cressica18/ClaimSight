"""
Documents API endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.api import deps
from app.models.claim import Claim
from app.models.document import Document
from app.models.enums import DocType, ExtractionStatus
from app.schemas.document import Document as DocumentSchema, DocumentSummary
from app.services import storage

router = APIRouter()

ALLOWED_DOC_TYPES = {"application/pdf", "image/jpeg", "image/png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.get(
    "/{id}/documents",
    response_model=list[DocumentSummary],
    summary="List documents for a claim",
)
def list_documents(
    id: int,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Return all documents attached to a claim, newest first.

    Phase 9: drives the Document Viewer screen (`/claims/:id/documents`).
    The list is intentionally lightweight (DocumentSummary) so the UI can
    paint the tab strip without paying for `extracted_fields` payloads.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found.",
        )

    stmt = (
        select(Document)
        .where(Document.claim_id == id)
        .order_by(desc(Document.created_at))
    )
    documents = db.execute(stmt).scalars().all()
    return documents


@router.get(
    "/{id}/documents/{doc_id}",
    response_model=DocumentSchema,
    summary="Get a single document",
)
def get_document(
    id: int,
    doc_id: int,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Return one document by id, scoped to a claim.

    Phase 9: used by the Document Viewer to fetch the full record
    (including `extracted_fields` + `raw_confidence`) for the active tab.
    """
    document = db.get(Document, doc_id)
    if not document or document.claim_id != id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found for this claim.",
        )
    return document


@router.post(
    "/{id}/documents",
    response_model=DocumentSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a claim document"
)
def upload_document(
    id: int,
    file: UploadFile = File(...),
    doc_type: DocType = Form(...),
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Upload a document (claim form, policy, estimate, invoice) for a claim.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")
        
    if file.content_type not in ALLOWED_DOC_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {file.content_type}. Allowed types are: {ALLOWED_DOC_TYPES}"
        )
        
    # Check size (requires seeking, then rewinding)
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE} bytes."
        )
        
    # Save file
    file_path = storage.save_upload_file(file, id)
    
    # Create DB record
    document = Document(
        claim_id=id,
        doc_type=doc_type.value,
        file_path=file_path,
        extraction_status=ExtractionStatus.pending.value
    )
    
    try:
        db.add(document)
        db.commit()
        db.refresh(document)
    except Exception as e:
        db.rollback()
        storage.delete_file(file_path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        
    return document
