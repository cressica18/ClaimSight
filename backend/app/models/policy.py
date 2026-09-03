"""Policy model — blueprint entity: Policy(id, customer_id, vehicle_id, policy_number,
coverage_type, coverage_limit, deductible, start_date, end_date, status)."""

from datetime import date, datetime

from sqlalchemy import (
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Numeric,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.models.enums import CoverageType, PolicyStatus


class Policy(Base):
    __tablename__ = "policies"
    __table_args__ = (
        UniqueConstraint("policy_number", name="uq_policies_policy_number"),
        CheckConstraint("deductible >= 0", name="ck_policies_deductible_positive"),
        CheckConstraint(
            "coverage_limit > 0", name="ck_policies_coverage_limit_positive"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(
        ForeignKey("customers.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    vehicle_id: Mapped[int] = mapped_column(
        ForeignKey("vehicles.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    policy_number: Mapped[str] = mapped_column(String(100), nullable=False)
    coverage_type: Mapped[str] = mapped_column(
        String(50), nullable=False, default=CoverageType.comprehensive.value
    )
    coverage_limit: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    deductible: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False, default=0)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=PolicyStatus.active.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    customer: Mapped["Customer"] = relationship(  # noqa: F821
        "Customer", back_populates="policies"
    )
    vehicle: Mapped["Vehicle"] = relationship(  # noqa: F821
        "Vehicle", back_populates="policies"
    )
    claims: Mapped[list["Claim"]] = relationship(  # noqa: F821
        "Claim", back_populates="policy"
    )

    def __repr__(self) -> str:
        return f"<Policy id={self.id} number={self.policy_number!r}>"
