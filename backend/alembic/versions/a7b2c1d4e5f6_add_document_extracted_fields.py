"""add document extracted_fields

Revision ID: a7b2c1d4e5f6
Revises: 3c1c641e2c53
Create Date: 2026-09-02 12:00:00.000000

Adds a nullable JSON column `extracted_fields` to the `documents` table.
This column carries the structured fields produced by the document
intelligence pipeline (blueprint Section 4) and is consumed by the
consistency engine's R9 (document_field_conflict) rule.

Phase 6 of the implementation roadmap.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7b2c1d4e5f6'
down_revision: Union[str, Sequence[str], None] = '3c1c641e2c53'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('extracted_fields', sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('documents', 'extracted_fields')
