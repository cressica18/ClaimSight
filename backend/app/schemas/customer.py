"""Customer Pydantic schemas — request/response models for the Customer entity."""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class CustomerBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    email: EmailStr
    phone: str | None = Field(None, max_length=50)


class CustomerCreate(CustomerBase):
    """Request body for POST /customers."""
    pass


class CustomerUpdate(BaseModel):
    """Request body for PATCH /customers/{id} (all fields optional)."""
    name: str | None = Field(None, min_length=1, max_length=255)
    email: EmailStr | None = None
    phone: str | None = Field(None, max_length=50)


class Customer(CustomerBase):
    """Full customer response."""
    id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class CustomerSummary(BaseModel):
    """Lightweight customer view for list responses."""
    id: int
    name: str
    email: str

    model_config = {"from_attributes": True}
