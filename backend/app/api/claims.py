"""
Claims API endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import select, desc

from app.api import deps
from app.models.claim import Claim
from app.models.policy import Policy
from app.models.vehicle import Vehicle
from app.models.previous_claim import PreviousClaim
from app.models.risk_signal import RiskSignal
from app.models.investigation import Investigation
from app.models.enums import ClaimStatus, RiskBand, Recommendation
from app.schemas.claim import Claim as ClaimSchema, ClaimCreate, ClaimDetail, ClaimSummary, ClaimUpdate
from app.schemas.previous_claim import PreviousClaim as PreviousClaimSchema
from app.schemas.evidence import RiskSignalWithEvidence
from app.schemas.investigation import InvestigationSummary

router = APIRouter()


@router.post(
    "",
    response_model=ClaimSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new claim"
)
def create_claim(
    claim_in: ClaimCreate,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Create a new claim linked to a policy and vehicle.
    """
    # Validate policy exists
    policy = db.get(Policy, claim_in.policy_id)
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Policy not found."
        )

    # Validate vehicle exists
    vehicle = db.get(Vehicle, claim_in.vehicle_id)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found."
        )

    claim = Claim(
        claim_number=claim_in.claim_number,
        policy_id=claim_in.policy_id,
        vehicle_id=claim_in.vehicle_id,
        incident_date=claim_in.incident_date,
        reported_date=claim_in.reported_date,
        claimed_amount=claim_in.claimed_amount,
        status=ClaimStatus.pending.value
    )
    db.add(claim)
    
    try:
        db.commit()
        db.refresh(claim)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A claim with this number already exists for this policy."
        )
        
    return claim


@router.get(
    "",
    response_model=list[ClaimSummary],
    summary="List claims"
)
def list_claims(
    status: ClaimStatus | None = Query(None, description="Filter by claim status"),
    risk_band: RiskBand | None = Query(None, description="Filter by risk band"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    List claims with optional filtering by status and risk band.
    """
    stmt = select(Claim).order_by(desc(Claim.created_at))
    
    if status:
        stmt = stmt.where(Claim.status == status.value)
    if risk_band:
        stmt = stmt.where(Claim.risk_band == risk_band.value)
        
    stmt = stmt.offset(skip).limit(limit)
    claims = db.execute(stmt).scalars().all()
    return claims


@router.get(
    "/{id}",
    response_model=ClaimDetail,
    summary="Get claim details"
)
def get_claim(
    id: int,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Get full details of a specific claim.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Claim not found."
        )
    return claim


@router.get(
    "/{id}/previous-claims",
    response_model=list[PreviousClaimSchema],
    summary="Get previous claims"
)
def get_previous_claims(
    id: int,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Get linked previous claims for the vehicle/customer of a given claim.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")
        
    stmt = select(PreviousClaim).where(
        (PreviousClaim.customer_id == claim.policy.customer_id) |
        (PreviousClaim.vehicle_id == claim.vehicle_id)
    ).order_by(desc(PreviousClaim.incident_date))
    
    previous_claims = db.execute(stmt).scalars().all()
    return previous_claims


@router.get(
    "/{id}/evidence",
    response_model=list[RiskSignalWithEvidence],
    summary="Get risk signals and evidence"
)
def get_evidence(
    id: int,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Get all risk signals and their linked evidence for a claim.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")
        
    stmt = select(RiskSignal).where(RiskSignal.claim_id == id)
    signals = db.execute(stmt).scalars().all()
    return signals


@router.get(
    "/{id}/investigation",
    response_model=InvestigationSummary,
    summary="Get investigation summary"
)
def get_investigation(
    id: int,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Get the LLM investigation summary for a claim.

    Phase 9: `key_concerns` is derived at read time from the claim's
    `RiskSignal` rows. The model does not store a `key_concerns` column
    (the Gemini layer's output is prose + a recommendation), so we surface
    the same rule-firing descriptions the user sees on the Risk Signals
    screen. Each concern starts with the rule's `R<n>_<rule_id>` token so
    the frontend can cross-link to `/claims/{id}/signals`. The
    `disclaimer` field is the project-mandated constant.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")

    stmt = select(Investigation).where(Investigation.claim_id == id)
    investigation = db.execute(stmt).scalars().first()

    if not investigation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No investigation found for this claim."
        )

    if not investigation.summary_text:
        # 202 Accepted typically means processing hasn't finished,
        # but returning a custom response schema might be cleaner.
        # Following the blueprint strictly, return 202 if pending.
        raise HTTPException(
            status_code=status.HTTP_202_ACCEPTED,
            detail="Investigation is pending or has not generated a summary yet."
        )

    # Derive key_concerns from the persisted RiskSignal rows so the
    # Investigation Summary screen has something concrete to render.
    signal_stmt = select(RiskSignal).where(RiskSignal.claim_id == id)
    signals = db.execute(signal_stmt).scalars().all()
    key_concerns = [
        f"[{signal.rule_id}] {signal.description}"
        for signal in signals
    ]

    # Return structured investigation response
    return InvestigationSummary(
        summary=investigation.summary_text,
        key_concerns=key_concerns,
        recommendation=Recommendation(investigation.recommendation),
        model_version=investigation.model_version,
    )


from typing import Literal

class DecisionRequest(BaseModel):
    decision: Literal["approve", "deny", "investigate", "manual_review"]
    notes: str | None = None

@router.post(
    "/{id}/decision",
    response_model=ClaimSchema,
    summary="Record officer decision"
)
def record_decision(
    id: int,
    decision_in: DecisionRequest,
    db: Session = Depends(deps.get_db)
) -> Any:
    """
    Officer records final decision on a claim.
    """
    claim = db.get(Claim, id)
    if not claim:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Claim not found.")
        
    if claim.status == ClaimStatus.decided.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Claim has already been {claim.status}."
        )
        
    claim.status = ClaimStatus.decided.value

    # Phase 9: persist the officer's notes on the claim row itself. Nullable
    # column added in this phase; the Pydantic schema (Claim.decision_notes)
    # surfaces it back to the frontend via GET /claims/{id}.
    if decision_in.notes is not None:
        claim.decision_notes = decision_in.notes

    db.commit()
    db.refresh(claim)

    return claim
