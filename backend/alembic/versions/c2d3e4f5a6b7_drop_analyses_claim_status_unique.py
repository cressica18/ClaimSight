"""drop redundant uq_analyses_claim_status unique constraint

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-09-03 12:00:00.000000

Phase 12 — testing & reliability.

The previous model had a plain `UniqueConstraint("claim_id", "status",
name="uq_analyses_claim_status")` on the analyses table, declared as
a "safety net" for backends that don't honor the partial unique index
on `(claim_id) WHERE status='running'`. In practice the safety net
backfired: re-running the pipeline (which is the normal user flow —
clicking "Re-run analysis" on the Claim Analysis page) inserts a new
row with the same `claim_id` and the same terminal `status='completed'`,
and the constraint fires with `UNIQUE constraint failed: analyses.claim_id,
analyses.status`.

The partial unique index is the correct guard. A claim may have many
historical Analysis rows; only two simultaneous `running` rows for the
same claim are forbidden, and the partial index enforces that.

Phase 12 regression test: `test_pipeline_rerun_after_completion_is_idempotent`.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c2d3e4f5a6b7'
down_revision: Union[str, Sequence[str], None] = 'b1c2d3e4f5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make the migration idempotent: it is a no-op if the constraint
    # does not exist (which can happen if the analyses table was
    # created via Base.metadata.create_all rather than the previous
    # migration, in which case the model-level constraint is also
    # gone).
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = {c["name"] for c in inspector.get_unique_constraints("analyses")}
    if "uq_analyses_claim_status" not in existing:
        return

    # SQLite does not support DROP CONSTRAINT directly. We need to
    # rebuild the table without the constraint. The standard recipe is
    # to copy the rows, drop the table, and recreate it. Alembic
    # provides a portable helper via batch operations.
    with op.batch_alter_table('analyses', recreate='always') as batch_op:
        batch_op.drop_constraint(
            'uq_analyses_claim_status', type_='unique',
        )


def downgrade() -> None:
    with op.batch_alter_table('analyses', recreate='always') as batch_op:
        batch_op.create_unique_constraint(
            'uq_analyses_claim_status', ['claim_id', 'status'],
        )
