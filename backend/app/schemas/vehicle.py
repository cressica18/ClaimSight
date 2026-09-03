"""Vehicle Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class VehicleBase(BaseModel):
    make: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., min_length=1, max_length=100)
    year: int = Field(..., ge=1900, le=2100)
    vin: str | None = Field(None, max_length=17)
    plate_number: str | None = Field(None, max_length=20)


class VehicleCreate(VehicleBase):
    """Request body for creating a vehicle. customer_id comes from URL path."""
    customer_id: int


class VehicleUpdate(BaseModel):
    make: str | None = Field(None, max_length=100)
    model: str | None = Field(None, max_length=100)
    year: int | None = Field(None, ge=1900, le=2100)
    vin: str | None = Field(None, max_length=17)
    plate_number: str | None = Field(None, max_length=20)


class Vehicle(VehicleBase):
    """Full vehicle response."""
    id: int
    customer_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class VehicleSummary(BaseModel):
    """Lightweight vehicle view."""
    id: int
    make: str
    model: str
    year: int
    plate_number: str | None

    model_config = {"from_attributes": True}
