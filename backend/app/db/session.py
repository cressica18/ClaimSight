"""
Database session configuration (SQLAlchemy 2.0).

Phase 1: The engine and session factory are configured from the environment but
the application does NOT attempt to connect on startup. A live PostgreSQL
instance is not required until Phase 2.

Phase 2 will:
- Add Alembic migration support
- Call Base.metadata.create_all() or run migrations on startup
- Add a startup event to verify the connection
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

# Create engine (connection pool is lazy — no actual TCP connection until first use)
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,       # enables connection health checks
    echo=(settings.app_env == "development"),  # SQL logging in dev only
)

# Session factory — used by the get_db() dependency in app/api/deps.py
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models.

    All models in app/models/ must inherit from this class so that
    Alembic can auto-detect them for migration generation.
    """
    pass
