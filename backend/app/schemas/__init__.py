"""
Pydantic request/response schemas package.

All schemas are separated from SQLAlchemy models — never expose ORM models
directly as API contracts (blueprint Section 10 / Phase 3 convention).

Naming convention:
  <Entity>Base    — shared fields
  <Entity>Create  — POST request body
  <Entity>Update  — PATCH request body (all optional)
  <Entity>        — full response model
  <Entity>Summary — lightweight list-view response
"""

from app.schemas.customer import Customer, CustomerCreate, CustomerSummary, CustomerUpdate
from app.schemas.vehicle import Vehicle, VehicleCreate, VehicleSummary, VehicleUpdate
from app.schemas.policy import Policy, PolicyCreate, PolicySummary, PolicyUpdate
from app.schemas.claim import Claim, ClaimCreate, ClaimDetail, ClaimSummary, ClaimUpdate
from app.schemas.document import Document, DocumentCreate, DocumentSummary
from app.schemas.repair import RepairEstimate, RepairEstimateCreate, RepairItem, RepairItemCreate
from app.schemas.previous_claim import PreviousClaim, PreviousClaimCreate
from app.schemas.risk_signal import RiskSignal, RiskSignalCreate
from app.schemas.evidence import Evidence, EvidenceCreate, RiskSignalWithEvidence
from app.schemas.investigation import Investigation, InvestigationCreate, InvestigationSummary

__all__ = [
    "Customer", "CustomerCreate", "CustomerSummary", "CustomerUpdate",
    "Vehicle", "VehicleCreate", "VehicleSummary", "VehicleUpdate",
    "Policy", "PolicyCreate", "PolicySummary", "PolicyUpdate",
    "Claim", "ClaimCreate", "ClaimDetail", "ClaimSummary", "ClaimUpdate",
    "Document", "DocumentCreate", "DocumentSummary",
    "RepairEstimate", "RepairEstimateCreate", "RepairItem", "RepairItemCreate",
    "PreviousClaim", "PreviousClaimCreate",
    "RiskSignal", "RiskSignalCreate",
    "Evidence", "EvidenceCreate", "RiskSignalWithEvidence",
    "Investigation", "InvestigationCreate", "InvestigationSummary",
]
