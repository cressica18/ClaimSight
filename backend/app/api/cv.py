"""
CV analysis API endpoints.

Exposes:
  POST /claims/{id}/damages/{damage_id}/analyze
    - Takes an already-uploaded image (tracked via Damage row)
    - Runs actual CV inference using real trained checkpoint
    - Updates/creates Damage rows with results
    - Returns the analysis result

  POST /claims/{id}/analyze-images
    - Runs CV inference on ALL uploaded images for a claim
    - Returns list of analysis results
"""

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.api import deps
from app.models.claim import Claim
from app.models.damage import Damage

router = APIRouter()


# ─── Response schemas ─────────────────────────────────────────────────────────

class DamageTypeResult(BaseModel):
    label: str
    confidence: float


class SeverityResult(BaseModel):
    label: str
    confidence: float


class CVAnalysisResult(BaseModel):
    damage_id: int
    claim_id: int
    damage_types: list[DamageTypeResult]
    severity: SeverityResult
    low_confidence: bool
    source_image: str | None
    model_version: str
    timestamp: str | None
    error: str | None

    model_config = {"from_attributes": False}


class CVAnalysisBatchResult(BaseModel):
    claim_id: int
    analyzed: int
    results: list[CVAnalysisResult]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_region_ref(region_ref: str | None) -> dict:
    """Parse region_ref JSON if it contains metadata, else return raw path."""
    if not region_ref:
        return {"image_path": ""}
    try:
        return json.loads(region_ref)
    except json.JSONDecodeError:
        return {"image_path": region_ref}


def _run_cv_and_respond(
    claim_id: int,
    image_path: str,
    damage_id: int,
    db: Session,
    predictor=None,
) -> CVAnalysisResult:
    """Run inference and return a response object (does NOT commit)."""
    # Lazy import so torch is not required at startup
    try:
        from ml.inference.predictor import get_predictor, CVPrediction
    except ImportError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"CV module not available: {exc}",
        )

    if predictor is None:
        predictor = get_predictor()

    # Resolve the actual image path from region_ref
    parsed = _parse_region_ref(image_path)
    actual_image_path = parsed.get("image_path", image_path)

    try:
        result: CVPrediction = predictor.predict_from_path(actual_image_path)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CV model weights not found. Run training first.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV inference failed: {e}",
        )

    return CVAnalysisResult(
        damage_id=damage_id,
        claim_id=claim_id,
        damage_types=[DamageTypeResult(label=d.label, confidence=d.confidence) for d in result.damage_types],
        severity=SeverityResult(label=result.severity.label, confidence=result.severity.confidence),
        low_confidence=result.low_confidence,
        source_image=result.source_image,
        model_version=result.model_version,
        timestamp=result.timestamp,
        error=result.error,
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post(
    "/{id}/damages/{damage_id}/analyze",
    response_model=CVAnalysisResult,
    summary="Run CV analysis on a single uploaded image",
)
def analyze_single_image(
    id: int,
    damage_id: int,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Run CV inference on an already-uploaded image (Damage record must exist).
    Creates new Damage rows with the real prediction results.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")

    damage = db.get(Damage, damage_id)
    if not damage or damage.claim_id != id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Damage record not found for this claim.")

    if not damage.region_ref:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Damage record has no associated image path.")

    # Run CV inference using the service
    from app.services.cv_service import run_cv_on_image
    damages = run_cv_on_image(db, id, damage.region_ref)

    # Persist the new damage rows so they survive the request lifecycle.
    # Without this commit, the get_db() context manager rolls the transaction
    # back on exit and the analysis results are silently lost.
    db.commit()
    for d in damages:
        db.refresh(d)

    # Return the analysis result from the first created damage
    if damages and damages[0].damage_type != "cv_error":
        result = CVAnalysisResult(
            damage_id=damages[0].id,
            claim_id=id,
            damage_types=[DamageTypeResult(label=d.damage_type, confidence=d.confidence or 0.0) for d in damages],
            severity=SeverityResult(label=damages[0].severity or "unknown", confidence=0.0),
            low_confidence=False,
            source_image=damage.region_ref,
            model_version="claimsight_cv_v1",
            timestamp=None,
            error=None,
        )
        # Try to extract low_confidence and severity confidence from region_ref
        if damages[0].region_ref:
            try:
                meta = json.loads(damages[0].region_ref)
                result.low_confidence = meta.get("low_confidence", False)
                if damages[0].severity:
                    result.severity.confidence = meta.get("severity_confidence", 0.0)
                result.model_version = meta.get("model_version", "claimsight_cv_v1")
                result.timestamp = meta.get("timestamp")
            except json.JSONDecodeError:
                pass
        return result
    else:
        # Error case
        return CVAnalysisResult(
            damage_id=damage_id,
            claim_id=id,
            damage_types=[],
            severity=SeverityResult(label="unknown", confidence=0.0),
            low_confidence=True,
            source_image=damage.region_ref,
            model_version="claimsight_cv_v1",
            timestamp=None,
            error="CV inference failed",
        )


@router.post(
    "/{id}/analyze-images",
    response_model=CVAnalysisBatchResult,
    summary="Run CV analysis on all uploaded images for a claim",
)
def analyze_all_images(
    id: int,
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Run CV inference on all Damage rows with source='image' for this claim.
    Useful for batch analysis after multiple images are uploaded.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")

    stmt = select(Damage).where(
        Damage.claim_id == id,
        Damage.source == "image",
    )
    damages = db.execute(stmt).scalars().all()

    if not damages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No uploaded images found for this claim. Upload images first.",
        )

    results: list[CVAnalysisResult] = []
    for damage in damages:
        if not damage.region_ref:
            continue

        # Run CV inference using the service
        from app.services.cv_service import run_cv_on_image
        new_damages = run_cv_on_image(db, id, damage.region_ref)

        if new_damages and new_damages[0].damage_type != "cv_error":
            result = CVAnalysisResult(
                damage_id=new_damages[0].id,
                claim_id=id,
                damage_types=[DamageTypeResult(label=d.damage_type, confidence=d.confidence or 0.0) for d in new_damages],
                severity=SeverityResult(label=new_damages[0].severity or "unknown", confidence=0.0),
                low_confidence=False,
                source_image=damage.region_ref,
                model_version="claimsight_cv_v1",
                timestamp=None,
                error=None,
            )
            if new_damages[0].region_ref:
                try:
                    meta = json.loads(new_damages[0].region_ref)
                    result.low_confidence = meta.get("low_confidence", False)
                    if new_damages[0].severity:
                        result.severity.confidence = meta.get("severity_confidence", 0.0)
                    result.model_version = meta.get("model_version", "claimsight_cv_v1")
                    result.timestamp = meta.get("timestamp")
                except json.JSONDecodeError:
                    pass
        else:
            result = CVAnalysisResult(
                damage_id=damage.id,
                claim_id=id,
                damage_types=[],
                severity=SeverityResult(label="unknown", confidence=0.0),
                low_confidence=True,
                source_image=damage.region_ref,
                model_version="claimsight_cv_v1",
                timestamp=None,
                error="CV inference failed",
            )
        results.append(result)

    db.commit()

    return CVAnalysisBatchResult(
        claim_id=id,
        analyzed=len(results),
        results=results,
    )
