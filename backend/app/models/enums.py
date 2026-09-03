"""
Shared SQLAlchemy column types and Python enums used across models.

All Postgres-native ENUM types are defined here so that:
1. Alembic sees a single definition and doesn't create duplicates.
2. Models import from one canonical location.

Blueprint constraints (Section 9):
- RiskSignal.severity → Postgres ENUM
- Investigation.recommendation → Postgres ENUM
"""

import enum

from sqlalchemy import Enum as SAEnum


# ─── Python Enums (source of truth) ─────────────────────────────────────────


class ClaimStatus(str, enum.Enum):
    pending = "pending"
    analyzing = "analyzing"
    completed = "completed"
    analysis_failed = "analysis_failed"
    decided = "decided"


class RiskBand(str, enum.Enum):
    low = "Low"
    medium = "Medium"
    high = "High"


class DamageSource(str, enum.Enum):
    image = "image"
    claim_form = "claim_form"


class DamageSeverity(str, enum.Enum):
    minor = "minor"
    moderate = "moderate"
    severe = "severe"


class DamageType(str, enum.Enum):
    scratch = "scratch"
    dent = "dent"
    crack = "crack"
    shattered_glass = "shattered_glass"
    bumper_damage = "bumper_damage"
    panel_damage = "panel_damage"
    headlight_damage = "headlight_damage"
    no_damage = "no_damage"


class DocType(str, enum.Enum):
    claim_form = "claim_form"
    policy = "policy"
    estimate = "estimate"
    invoice = "invoice"
    previous_claim = "previous_claim"


class ExtractionStatus(str, enum.Enum):
    pending = "pending"
    completed = "completed"
    failed = "failed"


class RepairOperation(str, enum.Enum):
    replace = "replace"
    repair = "repair"
    paint = "paint"


class SignalSeverity(str, enum.Enum):
    low = "low"
    medium = "medium"
    high = "high"


class EvidenceType(str, enum.Enum):
    image = "image"
    document = "document"
    field = "field"
    computed = "computed"


class Recommendation(str, enum.Enum):
    normal = "normal"
    manual_review = "manual_review"
    investigate = "investigate"


class PolicyStatus(str, enum.Enum):
    active = "active"
    expired = "expired"
    cancelled = "cancelled"


class CoverageType(str, enum.Enum):
    comprehensive = "comprehensive"
    third_party = "third_party"
    collision = "collision"
    fire_theft = "fire_theft"


class AnalysisStatus(str, enum.Enum):
    """Phase 11 — state machine for one pipeline run on a claim.

    The orchestrator transitions pending → running → {completed | failed}.
    `failed` is terminal AND implies the claim is marked analysis_failed
    so the user can re-run from the UI. The frontend polls
    `GET /claims/{id}/analysis/{analysis_id}` and surfaces both states.
    """
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# ─── SQLAlchemy ENUM column types (create_constraint=True → Postgres ENUM) ──
# Used directly in model Column definitions.

signal_severity_type = SAEnum(
    SignalSeverity,
    name="signal_severity",
    create_constraint=True,
    native_enum=True,
)

recommendation_type = SAEnum(
    Recommendation,
    name="recommendation",
    create_constraint=True,
    native_enum=True,
)
