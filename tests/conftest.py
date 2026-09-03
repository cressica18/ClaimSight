"""
pytest configuration for ClaimSight backend tests.

Phase 2 strategy:
- Model/relationship tests use SQLite in-memory so they run without a live
  PostgreSQL instance (important for CI and developer setup).
- SQLite doesn't support native Postgres ENUMs, so we configure SQLAlchemy
  to use VARCHAR for those columns in tests via create_engine options.
- A separate `test_pg_connection.py` verifies the real PG connection and is
  skipped gracefully when PG is not available.
- The Phase 1 `module`-scoped TestClient fixture is preserved unchanged so
  existing tests continue to pass.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base


# ─── Phase 1 fixture (unchanged) ──────────────────────────────────────────────


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Synchronous HTTPX TestClient for the FastAPI application."""
    with TestClient(app) as c:
        yield c


# ─── Phase 2: SQLite in-memory test database ──────────────────────────────────

SQLITE_URL = "sqlite+pysqlite:///:memory:"


@pytest.fixture(scope="function")
def sqlite_engine():
    """Create a fresh SQLite in-memory engine for each test function."""
    engine = create_engine(
        SQLITE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Enable FK enforcement in SQLite (off by default)
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables (Postgres ENUMs become VARCHAR in SQLite)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(sqlite_engine):
    """Yield a SQLAlchemy session backed by the SQLite in-memory engine.

    Each test gets its own isolated session; nothing is committed to disk.
    Also overrides the FastAPI get_db dependency for API tests.
    """
    TestingSession = sessionmaker(
        autocommit=False, autoflush=False, bind=sqlite_engine
    )
    session = TestingSession()
    
    def override_get_db():
        try:
            yield session
        finally:
            pass # Session closed by the fixture
            
    from app.main import app
    from app.api.deps import get_db
    app.dependency_overrides[get_db] = override_get_db

    try:
        yield session
    finally:
        session.rollback()
        session.close()
        app.dependency_overrides.clear()
