"""PreviousClaim model — blueprint entity:
PreviousClaim(id, customer_id, vehicle_id, claim_number, incident_date,
              damage_summary, claimed_amount, overlap_score)

Stores historical claims used by the consistency engine (Rule R5) to detect
duplicate/overlapping damage across claims for the same vehicle/customer.
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Numeric, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class PreviousClaim(Base):
    __tablename__ = "previous_claims"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    vehicle_id: Mapped[int] = mapped_column(
        ForeignKey("vehicles.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,  # Blueprint Section 9: index on PreviousClaim.vehicle_id
    )
    claim_number: Mapped[str] = mapped_column(String(100), nullable=False)
    incident_date: Mapped[date] = mapped_column(Date, nullable=False)
    damage_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    claimed_amount: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    # overlap_score computed by consistency engine (R5); null until engine runs
    overlap_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    customer: Mapped["Customer"] = relationship(  # noqa: F821
        "Customer", back_populates="previous_claims"
    )
    vehicle: Mapped["Vehicle"] = relationship(  # noqa: F821
        "Vehicle", back_populates="previous_claims"
    )

    def __repr__(self) -> str:
        return f"<PreviousClaim id={self.id} claim_number={self.claim_number!r}>"
