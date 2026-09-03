"""
Policy API endpoints (Phase 9).

The original Phase 3 backend shipped the `Policy` model and schema but
never exposed an HTTP endpoint for policies. Phase 9 adds the minimum
needed to drive the "Pick existing policy" / "Create new policy" steps
of the New Claim screen: list + create.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.api import deps
from app.models.customer import Customer
from app.models.policy import Policy
from app.models.vehicle import Vehicle
from app.schemas.policy import (
    Policy as PolicySchema,
    PolicyCreate,
    PolicySummary,
)

router = APIRouter()


@router.get(
    "",
    response_model=list[PolicySummary],
    summary="List policies",
)
def list_policies(
    customer_id: int | None = Query(None, description="Filter by customer id"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Return policies, newest first. Optionally filtered by `customer_id`
    so the New Claim screen can show only that customer's policies.
    """
    stmt = select(Policy).order_by(Policy.created_at.desc())
    if customer_id is not None:
        stmt = stmt.where(Policy.customer_id == customer_id)
    stmt = stmt.offset(skip).limit(limit)
    policies = db.execute(stmt).scalars().all()
    return policies


@router.post(
    "",
    response_model=PolicySchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new policy",
)
def create_policy(
    policy_in: PolicyCreate,
    db: Session = Depends(deps.get_db),
) -> Any:
    """Create a new policy. Validates customer and vehicle existence."""
    customer = db.get(Customer, policy_in.customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )
    vehicle = db.get(Vehicle, policy_in.vehicle_id)
    if not vehicle:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found.",
        )

    policy = Policy(
        policy_number=policy_in.policy_number,
        customer_id=policy_in.customer_id,
        vehicle_id=policy_in.vehicle_id,
        coverage_type=policy_in.coverage_type.value,
        coverage_limit=policy_in.coverage_limit,
        deductible=policy_in.deductible,
        start_date=policy_in.start_date,
        end_date=policy_in.end_date,
        status=policy_in.status.value,
    )
    db.add(policy)

    try:
        db.commit()
        db.refresh(policy)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A policy with this number already exists.",
        )

    return policy
