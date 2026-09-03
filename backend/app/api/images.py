"""
Images API endpoints.
"""

import json
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from sqlalchemy import select
from pydantic import BaseModel

from app.api import deps
from app.models.claim import Claim
from app.models.damage import Damage
from app.services import storage

router = APIRouter()

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

class DamageResponse(BaseModel):
    """Schema for Damage response."""
    id: int
    claim_id: int
    source: str
    damage_type: str | None
    severity: str | None
    confidence: float | None
    region_ref: str | None

    model_config = {"from_attributes": True}


@router.get(
    "/{id}/images",
    response_model=list[DamageResponse],
    summary="List uploaded images for a claim"
)
def list_images(
    id: int,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    List all uploaded images (Damage records with source='image') for a claim.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")

    stmt = select(Damage).where(
        Damage.claim_id == id,
        Damage.source == "image",
    )
    damages = db.execute(stmt).scalars().all()

    return damages


@router.post(
    "/{id}/images",
    response_model=list[DamageResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Upload accident images"
)
def upload_images(
    id: int,
    files: list[UploadFile] = File(...),
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Upload accident image(s) for a claim.
    Creates Damage records with the image path stored in region_ref for later CV analysis.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")

    damages = []
    saved_paths = []

    try:
        for file in files:
            if file.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file type: {file.content_type}. Allowed types are: {ALLOWED_IMAGE_TYPES}"
                )

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
            saved_paths.append(file_path)

            # Create Damage record with image path stored in region_ref as JSON metadata
            damage = Damage(
                claim_id=id,
                source="image",
                damage_type="pending",
                severity="pending",
                region_ref=json.dumps({"image_path": file_path}),
            )
            db.add(damage)
            damages.append(damage)

        db.commit()
        for damage in damages:
            db.refresh(damage)

    except Exception as e:
        db.rollback()
        for path in saved_paths:
            storage.delete_file(path)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return damages
