"""RiskSignal model — blueprint entity:
RiskSignal(id, claim_id, rule_id, category, severity [low|medium|high], description, created_at)

Blueprint Section 9: severity → Postgres ENUM type (signal_severity).
Blueprint Section 9: index on RiskSignal.claim_id.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.models.enums import SignalSeverity, signal_severity_type


class RiskSignal(Base):
    __tablename__ = "risk_signals"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        index=True,  # Blueprint Section 9
    )
    # rule_id: e.g. "R1_unsupported_damage", "R4_excessive_repair_cost"
    rule_id: Mapped[str] = mapped_column(String(100), nullable=False)
    category: Mapped[str] = mapped_column(String(100), nullable=False)
    # severity: Postgres ENUM as per blueprint Section 9
    severity: Mapped[str] = mapped_column(
        signal_severity_type,
        nullable=False,
        default=SignalSeverity.medium.value,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship("Claim", back_populates="risk_signals")  # noqa: F821
    evidence: Mapped[list["Evidence"]] = relationship(  # noqa: F821
        "Evidence", back_populates="risk_signal", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<RiskSignal id={self.id} rule={self.rule_id!r} severity={self.severity!r}>"
