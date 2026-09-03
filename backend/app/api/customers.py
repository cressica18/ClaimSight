"""
Customer API endpoints.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.api import deps
from app.models.customer import Customer
from app.schemas.customer import (
    Customer as CustomerSchema,
    CustomerCreate,
    CustomerSummary,
)

router = APIRouter()


@router.get(
    "",
    response_model=list[CustomerSummary],
    summary="List customers",
)
def list_customers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Return customers, newest first.

    Phase 9: powers the "Pick existing customer" step in the New Claim
    screen. Lightweight summary (id + name + email) — no phone / no
    created_at. Capped at 1000 rows; sufficient for a prototype.
    """
    stmt = (
        select(Customer)
        .order_by(Customer.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    customers = db.execute(stmt).scalars().all()
    return customers


@router.post(
    "",
    response_model=CustomerSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new customer"
)
def create_customer(
    customer_in: CustomerCreate,
    db: Session = Depends(deps.get_db)
):
    """
    Create a new customer in the system.
    """
    customer = Customer(
        name=customer_in.name,
        email=customer_in.email,
        phone=customer_in.phone
    )
    db.add(customer)

    try:
        db.commit()
        db.refresh(customer)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A customer with this email already exists."
        )

    return customer
