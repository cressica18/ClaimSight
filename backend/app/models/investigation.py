"""Investigation model — blueprint entity:
Investigation(id, claim_id, summary_text, recommendation [normal|manual_review|investigate],
              generated_at, model_version)

Blueprint Section 9: recommendation → Postgres ENUM type.
1:1 with Claim.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base
from app.models.enums import Recommendation, recommendation_type


class Investigation(Base):
    __tablename__ = "investigations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,   # Enforces 1:1 with Claim at DB level
        index=True,
    )
    # summary_text is null until the Gemini layer (Phase 8) generates it
    summary_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    # recommendation: Postgres ENUM as per blueprint Section 9
    recommendation: Mapped[str] = mapped_column(
        recommendation_type,
        nullable=False,
        default=Recommendation.manual_review.value,
    )
    generated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # ─── Relationships ────────────────────────────────────────────────────────
    claim: Mapped["Claim"] = relationship("Claim", back_populates="investigation")  # noqa: F821

    def __repr__(self) -> str:
        return f"<Investigation id={self.id} claim_id={self.claim_id} rec={self.recommendation!r}>"
