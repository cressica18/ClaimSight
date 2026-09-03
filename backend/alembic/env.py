"""Alembic environment configuration for ClaimSight.

Key decisions:
- Uses app.core.config.settings for the database URL (single source of truth).
- Imports app.models so all table definitions are registered on Base.metadata
  before autogenerate runs.
- Supports both offline (SQL script generation) and online (live DB) migration modes.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# ─── Load our application settings and models ─────────────────────────────────
# Importing settings first ensures DATABASE_URL is available.
from app.core.config import settings  # noqa: E402

# Import all models so Base.metadata knows about all tables.
import app.models  # noqa: E402, F401

from app.db.session import Base  # noqa: E402

# ─── Alembic Config object ─────────────────────────────────────────────────────
config = context.config

# Override sqlalchemy.url from our pydantic settings (ignores alembic.ini value)
config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The MetaData object containing all our model definitions.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This generates SQL scripts without requiring a live database connection.
    Useful for reviewing the migration SQL before applying it.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        # Include schema-level object drops (e.g., ENUM types) on downgrade
        include_schemas=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_schemas=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
