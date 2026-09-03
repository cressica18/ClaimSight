"""Repair models — blueprint entities:
  RepairEstimate(id, claim_id, document_id, shop_name, total_cost, currency, issued_date)
  RepairItem(id, repair_estimate_id, part_name, operation, cost, labor_hours)
"""

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, ForeignKey, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class RepairEstimate(Base):
    __tablename__ = "repair_estimates"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # document_id links to the source document this estimate was extracted from.
    # Nullable because an estimate may be entered manually before a document exists.
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    shop_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    total_cost: Mapped[float | None] = mapped_column(Numeric(12, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="USD")
    issued_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship(  # noqa: F821
        "Claim", back_populates="repair_estimates"
    )
    document: Mapped["Document | None"] = relationship(  # noqa: F821
        "Document", back_populates="repair_estimates"
    )
    items: Mapped[list["RepairItem"]] = relationship(
        "RepairItem", back_populates="estimate", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<RepairEstimate id={self.id} total={self.total_cost} {self.currency}>"


class RepairItem(Base):
    __tablename__ = "repair_items"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    repair_estimate_id: Mapped[int] = mapped_column(
        ForeignKey("repair_estimates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    part_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # operation: "replace" | "repair" | "paint"
    operation: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cost: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    labor_hours: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    estimate: Mapped["RepairEstimate"] = relationship(
        "RepairEstimate", back_populates="items"
    )

    def __repr__(self) -> str:
        return f"<RepairItem id={self.id} part={self.part_name!r} op={self.operation!r}>"
