"""add claims.decision_notes

Revision ID: d3e4f5a6b7c8
Revises: c2d3e4f5a6b7
Create Date: 2026-09-03 13:00:00.000000

Phase 13 — final polish.

The Claim model gained a `decision_notes` column in Phase 9 (the
Decision Panel persists the officer's notes onto the Claim row).
The migration that should have added the column was never written,
so existing production databases do not have it. The Phase 9 test
`test_decision_notes_column_exists` was added later and has been
passing only because it uses SQLite in-memory and `Base.metadata.create_all`.

This migration adds the column to existing deployments.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd3e4f5a6b7c8'
down_revision: Union[str, Sequence[str], None] = 'c2d3e4f5a6b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Idempotent: skip if the column already exists (e.g. when the
    # claims table was created via Base.metadata.create_all).
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    cols = {c["name"] for c in inspector.get_columns("claims")}
    if "decision_notes" in cols:
        return
    op.add_column(
        'claims',
        sa.Column('decision_notes', sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('claims', 'decision_notes')
