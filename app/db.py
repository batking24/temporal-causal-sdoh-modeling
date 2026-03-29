"""
db.py — SQLAlchemy engine, session management, and schema initialization.

Uses SQLite for zero-setup local development.
All tables are defined in sql/schema.sql and executed on first init.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()

# ---------------------------------------------------------------------------
# Engine — SQLite with WAL mode for concurrent read performance
# ---------------------------------------------------------------------------
engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False},     # needed for SQLite + threads
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, connection_record):   # noqa: ANN001
    """Tune SQLite for analytical workloads."""
    if isinstance(dbapi_conn, sqlite3.Connection):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")         # write-ahead logging
        cursor.execute("PRAGMA synchronous=NORMAL;")       # faster writes
        cursor.execute("PRAGMA foreign_keys=ON;")          # enforce FKs
        cursor.execute("PRAGMA cache_size=-64000;")        # 64 MB page cache
        cursor.execute("PRAGMA temp_store=MEMORY;")        # temp tables in RAM
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context-managed database session.

    Usage::

        with get_db_session() as session:
            session.execute(text("SELECT 1"))
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database session error — rolled back")
        raise
    finally:
        session.close()


def get_raw_connection() -> sqlite3.Connection:
    """Return a raw sqlite3 connection for bulk inserts (much faster)."""
    import os
    os.makedirs(Path(settings.DB_PATH).parent, exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------
_SCHEMA_PATH = Path(settings.PROJECT_ROOT) / "sql" / "schema.sql"


def init_db() -> None:
    """Execute sql/schema.sql against the database to create all tables."""
    if not _SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {_SCHEMA_PATH}")

    ddl = _SCHEMA_PATH.read_text(encoding="utf-8")

    conn = get_raw_connection()
    try:
        conn.executescript(ddl)
        conn.commit()
        logger.info("Database schema initialized from %s", _SCHEMA_PATH)
    finally:
        conn.close()


def verify_db() -> dict:
    """
    Quick health check — returns dict of table names → row counts.
    Useful for verifying that ingestion ran correctly.
    """
    with get_db_session() as session:
        tables_result = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        )
        tables = [row[0] for row in tables_result]

        counts = {}
        for table in tables:
            result = session.execute(text(f"SELECT COUNT(*) FROM [{table}]"))  # noqa: S608
            counts[table] = result.scalar()

    return counts


# ---------------------------------------------------------------------------
# CLI entry point — run `python -m app.db` to init + verify
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)
    logger.info("Initializing database at %s", settings.DB_PATH)
    init_db()
    counts = verify_db()
    print("\n=== Database Health Check ===")
    print(json.dumps(counts, indent=2))
    print(f"\nTotal tables: {len(counts)}")
    print("✅ Database ready.")
