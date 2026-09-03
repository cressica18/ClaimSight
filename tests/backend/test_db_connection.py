"""
Test actual PostgreSQL database connection.

This test connects to the live Postgres database to verify that:
1. The credentials in settings.database_url are correct.
2. The database is reachable.
3. The Alembic migrations were successfully applied (tables exist).
"""

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from app.core.config import settings


def test_postgresql_connection():
    """Verify live PostgreSQL connection and schema."""
    engine = create_engine(settings.database_url)
    try:
        with engine.connect() as conn:
            # Check connection works
            result = conn.execute(text("SELECT 1")).scalar()
            assert result == 1

            # Check that a core table (e.g. customers) exists
            tables_result = conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            )
            tables = {row[0] for row in tables_result}
            
            assert "customers" in tables
            assert "claims" in tables
            assert "alembic_version" in tables
            
    except OperationalError as e:
        pytest.fail(f"Could not connect to PostgreSQL: {e}")
