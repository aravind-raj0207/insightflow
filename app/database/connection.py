"""Two database connections - this is a key part of the security design.

admin_engine    : can write. Used for uploads and the app's own metadata tables.
readonly_engine : can only SELECT from the user_data schema. The ONLY engine
                  that AI-generated SQL is ever executed with.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings

settings = get_settings()

admin_engine: Engine = create_engine(settings.admin_db_url, pool_pre_ping=True)
readonly_engine: Engine = create_engine(settings.readonly_db_url, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=admin_engine, expire_on_commit=False)


def check_database(engine: Engine) -> bool:
    """True if we can connect and run a trivial query."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def describe_db_error(engine: Engine) -> str | None:
    """Returns a short, human-readable reason if the database cannot be used, else None."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return None
    except Exception as exc:
        reason = str(getattr(exc, "orig", exc)).strip().splitlines()[0]
        return reason
