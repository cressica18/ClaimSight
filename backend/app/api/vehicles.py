"""
Vehicle API endpoints (Phase 9).

Like `policies.py`, the Vehicle model existed but no HTTP endpoints
were wired. Phase 9 adds list + create to back the New Claim screen.
"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.api import deps
from app.models.customer import Customer
from app.models.vehicle import Vehicle
from app.schemas.vehicle import (
    Vehicle as VehicleSchema,
    VehicleCreate,
    VehicleSummary,
)

router = APIRouter()


@router.get(
    "",
    response_model=list[VehicleSummary],
    summary="List vehicles",
)
def list_vehicles(
    customer_id: int | None = Query(None, description="Filter by customer id"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Return vehicles, newest first. Optionally filtered by `customer_id`.
    """
    stmt = select(Vehicle).order_by(Vehicle.created_at.desc())
    if customer_id is not None:
        stmt = stmt.where(Vehicle.customer_id == customer_id)
    stmt = stmt.offset(skip).limit(limit)
    vehicles = db.execute(stmt).scalars().all()
    return vehicles


@router.post(
    "",
    response_model=VehicleSchema,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new vehicle",
)
def create_vehicle(
    vehicle_in: VehicleCreate,
    db: Session = Depends(deps.get_db),
) -> Any:
    """Create a new vehicle. Validates customer existence."""
    customer = db.get(Customer, vehicle_in.customer_id)
    if not customer:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Customer not found.",
        )

    vehicle = Vehicle(
        customer_id=vehicle_in.customer_id,
        make=vehicle_in.make,
        model=vehicle_in.model,
        year=vehicle_in.year,
        vin=vehicle_in.vin,
        plate_number=vehicle_in.plate_number,
    )
    db.add(vehicle)

    try:
        db.commit()
        db.refresh(vehicle)
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A vehicle with this VIN already exists.",
        )

    return vehicle
