"""Document model — blueprint entity: Document(id, claim_id, doc_type, file_path,
extraction_status, raw_confidence, extracted_fields)."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # doc_type: "claim_form" | "policy" | "estimate" | "invoice" | "previous_claim"
    doc_type: Mapped[str] = mapped_column(String(30), nullable=False)
    # Relative path inside data/uploads/{claim_id}/
    file_path: Mapped[str] = mapped_column(String(1000), nullable=False)
    # extraction_status: "pending" | "completed" | "failed"
    extraction_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending"
    )
    # Overall confidence from extraction (0.0–1.0); null until extraction runs
    raw_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Structured fields extracted from the document (e.g. policy_number,
    # plate_number, vin). Populated by the document-intelligence pipeline
    # (blueprint Section 4). Stored as JSON; nullable for backwards
    # compatibility with documents uploaded before extraction existed.
    extracted_fields: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship("Claim", back_populates="documents")  # noqa: F821
    repair_estimates: Mapped[list["RepairEstimate"]] = relationship(  # noqa: F821
        "RepairEstimate", back_populates="document"
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} type={self.doc_type!r} status={self.extraction_status!r}>"
