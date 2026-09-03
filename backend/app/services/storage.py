"""
Local file storage service.

Handles secure storage of uploaded documents and images.
In later phases, this could be swapped out for S3/GCS.
"""

import os
import shutil
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings


def get_claim_upload_dir(claim_id: int) -> Path:
    """Get the upload directory for a specific claim."""
    base_dir = Path(settings.upload_dir)
    # Convert absolute paths safely if upload_dir is relative
    if not base_dir.is_absolute():
        base_dir = Path(os.getcwd()) / base_dir
    claim_dir = base_dir / str(claim_id)
    return claim_dir


def save_upload_file(upload_file: UploadFile, claim_id: int) -> str:
    """
    Save an uploaded file securely to local storage.
    
    Returns the relative path to store in the database (e.g., "uploads/{claim_id}/{uuid}-{filename}").
    """
    claim_dir = get_claim_upload_dir(claim_id)
    claim_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate safe filename
    file_uuid = uuid.uuid4().hex[:8]
    safe_filename = f"{file_uuid}-{upload_file.filename}"
    
    # Secure against path traversal
    safe_filename = os.path.basename(safe_filename)
    
    file_path = claim_dir / safe_filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
        
    return f"uploads/{claim_id}/{safe_filename}"


def delete_file(relative_path: str):
    """
    Delete a file from local storage given its relative path.
    Useful for cleanup on database transaction failure.
    """
    if not relative_path.startswith("uploads/"):
        return
        
    parts = relative_path.split("/")
    if len(parts) < 3:
        return
        
    claim_id_str = parts[1]
    filename = parts[2]
    
    base_dir = Path(settings.upload_dir)
    if not base_dir.is_absolute():
        base_dir = Path(os.getcwd()) / base_dir
        
    file_path = base_dir / claim_id_str / filename
    if file_path.exists():
        os.remove(file_path)
