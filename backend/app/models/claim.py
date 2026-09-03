"""Claim model — blueprint entity: Claim(id, policy_id, vehicle_id, claim_number,
incident_date, reported_date, claimed_amount, status, risk_band, risk_score, created_at).

Blueprint Section 9 constraints:
- Claim.risk_score CHECK between 0–100
- Unique (policy_id, incident_date, claim_number) per Section 13 duplicate guard
- Indexes on policy_id, vehicle_id, status
"""

from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.models.enums import ClaimStatus, RiskBand


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        UniqueConstraint(
            "policy_id", "claim_number",
            name="uq_claims_policy_claim_number",
        ),
        CheckConstraint(
            "risk_score IS NULL OR (risk_score >= 0 AND risk_score <= 100)",
            name="ck_claims_risk_score_range",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    policy_id: Mapped[int] = mapped_column(
        ForeignKey("policies.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    vehicle_id: Mapped[int] = mapped_column(
        ForeignKey("vehicles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    claim_number: Mapped[str] = mapped_column(String(100), nullable=False)
    incident_date: Mapped[date] = mapped_column(Date, nullable=False)
    reported_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    claimed_amount: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default=ClaimStatus.pending.value,
        index=True,
    )
    risk_band: Mapped[str | None] = mapped_column(String(10), nullable=True)
    risk_score: Mapped[float | None] = mapped_column(Numeric(5, 2), nullable=True)
    # Officer decision notes (Phase 9). Populated by POST /claims/{id}/decision.
    # Stored on the Claim row directly so the Decision Panel can read it back
    # via GET /claims/{id} without a join. Nullable for backwards compatibility
    # with claims decided before this column existed.
    decision_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    policy: Mapped["Policy"] = relationship(  # noqa: F821
        "Policy", back_populates="claims"
    )
    vehicle: Mapped["Vehicle"] = relationship(  # noqa: F821
        "Vehicle", back_populates="claims"
    )
    accident: Mapped["Accident | None"] = relationship(  # noqa: F821
        "Accident", back_populates="claim", uselist=False, cascade="all, delete-orphan"
    )
    damages: Mapped[list["Damage"]] = relationship(  # noqa: F821
        "Damage", back_populates="claim", cascade="all, delete-orphan"
    )
    documents: Mapped[list["Document"]] = relationship(  # noqa: F821
        "Document", back_populates="claim", cascade="all, delete-orphan"
    )
    repair_estimates: Mapped[list["RepairEstimate"]] = relationship(  # noqa: F821
        "RepairEstimate", back_populates="claim", cascade="all, delete-orphan"
    )
    risk_signals: Mapped[list["RiskSignal"]] = relationship(  # noqa: F821
        "RiskSignal", back_populates="claim", cascade="all, delete-orphan"
    )
    investigation: Mapped["Investigation | None"] = relationship(  # noqa: F821
        "Investigation",
        back_populates="claim",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Claim id={self.id} number={self.claim_number!r} status={self.status!r}>"
