"""Damage model — blueprint entity: Damage(id, claim_id, source, damage_type, severity, confidence, region_ref)."""

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Damage(Base):
    __tablename__ = "damages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # source: "image" | "claim_form"  — stored as varchar, validated in schema
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    # damage_type: one of the DamageType enum values
    damage_type: Mapped[str] = mapped_column(String(50), nullable=False)
    # severity: "minor" | "moderate" | "severe" — nullable (not always known)
    severity: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # confidence 0.0–1.0 from CV model; null if source is claim_form
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # region_ref: free-form reference to image region / bounding box description
    region_ref: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship("Claim", back_populates="damages")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Damage id={self.id} type={self.damage_type!r} severity={self.severity!r}>"
