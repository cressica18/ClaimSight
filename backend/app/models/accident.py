"""Accident model — blueprint entity: Accident(id, claim_id, description, location, incident_type)."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Accident(Base):
    __tablename__ = "accidents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    location: Mapped[str | None] = mapped_column(String(500), nullable=True)
    incident_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship("Claim", back_populates="accident")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Accident id={self.id} claim_id={self.claim_id}>"
