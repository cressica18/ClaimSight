"""Evidence model — blueprint entity:
Evidence(id, risk_signal_id, evidence_type [image|document|field|computed],
         reference, detail_json)

Blueprint Section 9: index on Evidence.risk_signal_id.
Blueprint Section 8: every RiskSignal must have ≥1 Evidence row.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Evidence(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    risk_signal_id: Mapped[int] = mapped_column(
        ForeignKey("risk_signals.id", ondelete="CASCADE"),
        nullable=False,
        index=True,  # Blueprint Section 9
    )
    # evidence_type: "image" | "document" | "field" | "computed"
    evidence_type: Mapped[str] = mapped_column(String(20), nullable=False)
    # reference: image_id, document_id, or field name depending on type
    reference: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # detail_json: structured evidence payload (bounding box, field values, calc inputs)
    detail_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    risk_signal: Mapped["RiskSignal"] = relationship(  # noqa: F821
        "RiskSignal", back_populates="evidence"
    )

    def __repr__(self) -> str:
        return f"<Evidence id={self.id} type={self.evidence_type!r} signal={self.risk_signal_id}>"
