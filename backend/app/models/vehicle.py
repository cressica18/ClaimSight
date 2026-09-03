"""Vehicle model — blueprint entity: Vehicle(id, customer_id, make, model, year, vin, plate_number)."""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Vehicle(Base):
    __tablename__ = "vehicles"
    __table_args__ = (
        # A VIN is globally unique; plate numbers are unique per registration region
        # but we enforce uniqueness at the application level on VIN only.
        UniqueConstraint("vin", name="uq_vehicles_vin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="CASCADE"), nullable=False, index=True
    )
    make: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    vin: Mapped[str | None] = mapped_column(String(17), nullable=True)
    plate_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    customer: Mapped["Customer"] = relationship(  # noqa: F821
        "Customer", back_populates="vehicles"
    )
    policies: Mapped[list["Policy"]] = relationship(  # noqa: F821
        "Policy", back_populates="vehicle"
    )
    claims: Mapped[list["Claim"]] = relationship(  # noqa: F821
        "Claim", back_populates="vehicle"
    )
    previous_claims: Mapped[list["PreviousClaim"]] = relationship(  # noqa: F821
        "PreviousClaim", back_populates="vehicle"
    )

    def __repr__(self) -> str:
        return f"<Vehicle id={self.id} {self.year} {self.make} {self.model}>"
