"""
SQLAlchemy ORM models package.

IMPORTANT: All model classes must be imported here so that:
1. Alembic's env.py (which imports Base from app.db.session) can auto-detect
   all table definitions via Base.metadata.
2. SQLAlchemy's relationship() resolution works when models reference each other
   by string name.

Import order follows FK dependency (parents before children), though SQLAlchemy
handles the actual resolution at runtime.
"""

from app.models.customer import Customer
from app.models.vehicle import Vehicle
from app.models.policy import Policy
from app.models.claim import Claim
from app.models.accident import Accident
from app.models.damage import Damage
from app.models.document import Document
from app.models.repair import RepairEstimate, RepairItem
from app.models.previous_claim import PreviousClaim
from app.models.risk_signal import RiskSignal
from app.models.evidence import Evidence
from app.models.investigation import Investigation
from app.models.analysis import Analysis
from app.models.enums import (
    ClaimStatus,
    RiskBand,
    DamageSource,
    DamageSeverity,
    DamageType,
    DocType,
    ExtractionStatus,
    RepairOperation,
    SignalSeverity,
    EvidenceType,
    Recommendation,
    PolicyStatus,
    CoverageType,
    AnalysisStatus,
)

__all__ = [
    # Models
    "Customer",
    "Vehicle",
    "Policy",
    "Claim",
    "Accident",
    "Damage",
    "Document",
    "RepairEstimate",
    "RepairItem",
    "PreviousClaim",
    "RiskSignal",
    "Evidence",
    "Investigation",
    "Analysis",
    # Enums
    "ClaimStatus",
    "RiskBand",
    "DamageSource",
    "DamageSeverity",
    "DamageType",
    "DocType",
    "ExtractionStatus",
    "RepairOperation",
    "SignalSeverity",
    "EvidenceType",
    "Recommendation",
    "PolicyStatus",
    "CoverageType",
    "AnalysisStatus",
]
