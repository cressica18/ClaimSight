"""Customer model — blueprint entity: Customer(id, name, email, phone, created_at)."""

from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    vehicles: Mapped[list["Vehicle"]] = relationship(  # noqa: F821
        "Vehicle", back_populates="customer", cascade="all, delete-orphan"
    )
    policies: Mapped[list["Policy"]] = relationship(  # noqa: F821
        "Policy", back_populates="customer"
    )
    previous_claims: Mapped[list["PreviousClaim"]] = relationship(  # noqa: F821
        "PreviousClaim", back_populates="customer"
    )

    def __repr__(self) -> str:
        return f"<Customer id={self.id} email={self.email!r}>"
