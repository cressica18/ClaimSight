"""add analyses table

Revision ID: b1c2d3e4f5a6
Revises: a7b2c1d4e5f6
Create Date: 2026-09-02 13:00:00.000000

Phase 11 — Full Analysis Pipeline Integration (blueprint Section 12).

Adds the `analyses` table backing POST /claims/{id}/analyze and
GET /claims/{id}/analysis/{analysis_id}. One row per pipeline run on
a claim. The partial unique index enforces "at most one running
analysis per claim" at the DB level.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b1c2d3e4f5a6'
down_revision: Union[str, Sequence[str], None] = 'a7b2c1d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'analyses',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('claim_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.Column('current_step', sa.String(length=30), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['claim_id'], ['claims.id'], ondelete='CASCADE'),
    )
    op.create_index('ix_analyses_claim_id', 'analyses', ['claim_id'])

    # Partial unique index: at most one running analysis per claim.
    # SQLite supports CREATE UNIQUE INDEX ... WHERE since 3.8.0, and
    # PostgreSQL has it as a standard feature. The application's
    # in-process lock is the primary guard; this index is the safety
    # net for multi-process deployments.
    op.execute(
        "CREATE UNIQUE INDEX uq_analyses_one_running_per_claim "
        "ON analyses (claim_id) WHERE status = 'running'"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_analyses_one_running_per_claim")
    op.drop_index('ix_analyses_claim_id', table_name='analyses')
    op.drop_table('analyses')
