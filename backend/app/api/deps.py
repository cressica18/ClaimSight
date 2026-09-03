"""
Shared FastAPI dependencies.

This module will grow in Phase 2+ to provide:
- get_db() — yields a SQLAlchemy database session per request
- get_current_user() — auth dependency (if auth is added)
"""

from typing import Generator

from sqlalchemy.orm import Session

from app.db.session import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and ensure it is closed after the request.

    Usage in route handlers:
        db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
