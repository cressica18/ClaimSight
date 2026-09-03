"""Policy Pydantic schemas."""

from datetime import date, datetime

from pydantic import BaseModel, Field, model_validator

from app.models.enums import CoverageType, PolicyStatus


class PolicyBase(BaseModel):
    policy_number: str = Field(..., min_length=1, max_length=100)
    coverage_type: CoverageType = CoverageType.comprehensive
    coverage_limit: float = Field(..., gt=0)
    deductible: float = Field(0.0, ge=0)
    start_date: date
    end_date: date
    status: PolicyStatus = PolicyStatus.active

    @model_validator(mode="after")
    def validate_date_range(self) -> "PolicyBase":
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class PolicyCreate(PolicyBase):
    """Request body for creating a policy."""
    customer_id: int
    vehicle_id: int


class PolicyUpdate(BaseModel):
    coverage_limit: float | None = Field(None, gt=0)
    deductible: float | None = Field(None, ge=0)
    end_date: date | None = None
    status: PolicyStatus | None = None


class Policy(PolicyBase):
    """Full policy response."""
    id: int
    customer_id: int
    vehicle_id: int
    created_at: datetime

    model_config = {"from_attributes": True}


class PolicySummary(BaseModel):
    """Lightweight policy view."""
    id: int
    policy_number: str
    coverage_type: str
    status: str
    end_date: date

    model_config = {"from_attributes": True}
