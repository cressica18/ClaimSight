"""Analysis model — Phase 11.

Represents one execution of the /analyze pipeline on a claim. The
blueprint (Section 12) requires POST /claims/{id}/analyze to return
202 + analysis_id, and GET /claims/{id}/analysis/{analysis_id} to
return the current status of that run. This model is the storage
backing those two endpoints.

State machine (mirrors app.models.enums.AnalysisStatus):
    pending → running → {completed | failed}

A claim may have many historical Analysis rows; only one with
status='running' is allowed at a time. The DB-level guarantee is a
partial unique index on (claim_id) WHERE status='running', added in
the Alembic migration. SQLite (test) supports the same syntax.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (
        # The partial unique index `uq_analyses_one_running_per_claim`
        # is created in the Alembic migration. We do NOT also declare a
        # plain UniqueConstraint on (claim_id, status) because that
        # would block re-running the pipeline: a second analysis for
        # the same claim would also reach `completed` and would violate
        # the constraint. The partial index is the only DB-level guard,
        # enforced via `CREATE UNIQUE INDEX ... WHERE status='running'`.
        # (Phase 12 fix — the prior UniqueConstraint incorrectly fired
        # on the second completed run; the comment that called it a
        # "no-op" was wrong.)
        Index("ix_analyses_claim_id", "claim_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    claim_id: Mapped[int] = mapped_column(
        ForeignKey("claims.id", ondelete="CASCADE"),
        nullable=False,
    )
    # status: "pending" | "running" | "completed" | "failed"
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # current_step: best-effort marker, set by the orchestrator before each
    # pipeline step. Values: "cv" | "documents" | "context" | "rules" |
    # "risk" | "evidence" | "investigation" | None (between steps or terminal).
    current_step: Mapped[str | None] = mapped_column(String(30), nullable=True)
    # error_message: populated only on status=failed. Truncated to 2000 chars
    # by the orchestrator to keep the column bounded.
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    def __repr__(self) -> str:
        return f"<Analysis id={self.id} claim={self.claim_id} status={self.status!r}>"
