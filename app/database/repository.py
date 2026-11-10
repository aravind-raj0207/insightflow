"""All database reads/writes for the app live here, so other code never builds SQL for metadata."""
import logging
import uuid

import pandas as pd
from sqlalchemy import BigInteger, Boolean, DateTime, Float, Text, select, text

from app.config import get_settings
from app.database.connection import SessionLocal, admin_engine
from app.database.models import (
    Base, Conversation, Dataset, DatasetColumn, Message, QueryLog,
)
from app.utils.errors import DatasetNotFoundError, InsightFlowError

logger = logging.getLogger(__name__)
USER_DATA_SCHEMA = "user_data"

_SQLALCHEMY_TYPES = {
    "BIGINT": BigInteger,
    "DOUBLE PRECISION": Float,
    "TIMESTAMP": DateTime,
    "BOOLEAN": Boolean,
    "TEXT": Text,
}


def ensure_readonly_role() -> None:
    """Make sure the read-only database user exists and can only SELECT from user_data.

    db/init.sql normally does this, but it only runs when the Docker volume is first created.
    Doing it here as well means the app also works on an existing database. Needs an admin user
    that is allowed to create roles (the docker-compose admin is a superuser).
    """
    settings = get_settings()
    role = settings.db_readonly_user.replace('"', "")
    password = settings.db_readonly_password.replace("'", "''").replace("%", "%%")
    statements = [
        f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') "
        f"THEN CREATE ROLE \"{role}\" LOGIN; END IF; END $$",
        f"ALTER ROLE \"{role}\" WITH LOGIN PASSWORD '{password}'",          # keeps it in sync with .env
        f'GRANT CONNECT ON DATABASE "{settings.db_name}" TO "{role}"',
        f'GRANT USAGE ON SCHEMA {USER_DATA_SCHEMA} TO "{role}"',
        f'GRANT SELECT ON ALL TABLES IN SCHEMA {USER_DATA_SCHEMA} TO "{role}"',
        f'ALTER DEFAULT PRIVILEGES IN SCHEMA {USER_DATA_SCHEMA} GRANT SELECT ON TABLES TO "{role}"',
        f'REVOKE ALL ON SCHEMA app FROM "{role}"',
    ]
    try:
        with admin_engine.begin() as conn:
            for statement in statements:
                conn.exec_driver_sql(statement)
    except Exception as exc:
        logger.warning("Could not set up the read-only database user automatically: %s", exc)


def init_db() -> None:
    """Create schemas, metadata tables and the read-only user if they do not exist yet."""
    with admin_engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS app"))
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {USER_DATA_SCHEMA}"))
    Base.metadata.create_all(admin_engine)
    ensure_readonly_role()
    logger.info("Database initialised")


# ---------------------------------------------------------------- datasets

def _dataset_to_dict(dataset: Dataset, include_columns: bool = True) -> dict:
    result = {
        "id": dataset.id,
        "name": dataset.name,
        "table_name": dataset.table_name,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
    }
    if include_columns:
        result["columns"] = [
            {
                "name": c.name, "original_name": c.original_name, "kind": c.kind,
                "sql_type": c.sql_type, "missing_count": c.missing_count,
                "unique_count": c.unique_count, "stats": c.stats, "top_values": c.top_values,
            }
            for c in dataset.columns
        ]
        result["profile"] = dataset.profile
    return result


def create_dataset(df: pd.DataFrame, display_name: str, profile: dict) -> dict:
    """Store the DataFrame as a new table in user_data and save its metadata."""
    settings = get_settings()
    table_name = f"ds_{uuid.uuid4().hex[:10]}"
    dtypes = {col["name"]: _SQLALCHEMY_TYPES[col["sql_type"]] for col in profile["columns"]}

    try:
        df.to_sql(
            table_name, admin_engine, schema=USER_DATA_SCHEMA, index=False,
            dtype=dtypes, chunksize=2000, method="multi",
        )
        with admin_engine.begin() as conn:
            # Make sure the read-only role can read this table (and nothing else)
            conn.execute(text(
                f'GRANT SELECT ON {USER_DATA_SCHEMA}."{table_name}" TO "{settings.db_readonly_user}"'
            ))
    except Exception as exc:
        logger.exception("Failed to store dataset table %s", table_name)
        _drop_table_quietly(table_name)
        raise InsightFlowError("The dataset could not be stored in the database.", str(exc))

    try:
        with SessionLocal() as session:
            dataset = Dataset(
                name=display_name, table_name=table_name,
                row_count=profile["row_count"], column_count=profile["column_count"],
                profile=profile,
            )
            for position, col in enumerate(profile["columns"]):
                dataset.columns.append(DatasetColumn(
                    position=position, name=col["name"],
                    original_name=col.get("original_name", col["name"]),
                    kind=col["kind"], sql_type=col["sql_type"],
                    missing_count=col["missing_count"], unique_count=col["unique_count"],
                    stats=col["stats"], top_values=col["top_values"],
                ))
            session.add(dataset)
            session.commit()
            return _dataset_to_dict(dataset)
    except Exception as exc:
        logger.exception("Failed to save dataset metadata")
        _drop_table_quietly(table_name)
        raise InsightFlowError("The dataset metadata could not be saved.", str(exc))


def get_dataset(dataset_id: int) -> dict:
    with SessionLocal() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset {dataset_id} was not found.")
        return _dataset_to_dict(dataset)


def list_datasets() -> list[dict]:
    with SessionLocal() as session:
        rows = session.scalars(select(Dataset).order_by(Dataset.id.desc())).all()
        return [_dataset_to_dict(d, include_columns=False) for d in rows]


def delete_dataset(dataset_id: int) -> None:
    with SessionLocal() as session:
        dataset = session.get(Dataset, dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset {dataset_id} was not found.")
        table_name = dataset.table_name
        session.delete(dataset)
        session.commit()
    _drop_table_quietly(table_name)


def _drop_table_quietly(table_name: str) -> None:
    try:
        with admin_engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS {USER_DATA_SCHEMA}."{table_name}"'))
    except Exception:
        logger.exception("Could not drop table %s", table_name)


# ----------------------------------------------------------- conversations

def create_conversation(dataset_id: int) -> int:
    with SessionLocal() as session:
        conversation = Conversation(dataset_id=dataset_id)
        session.add(conversation)
        session.commit()
        return conversation.id


def get_conversation_dataset_id(conversation_id: int) -> int | None:
    with SessionLocal() as session:
        conversation = session.get(Conversation, conversation_id)
        return conversation.dataset_id if conversation else None


def add_message(conversation_id: int, role: str, content: str, *, resolved_question: str | None = None,
                sql: str | None = None, chart_spec: dict | None = None, status: str | None = None) -> None:
    with SessionLocal() as session:
        session.add(Message(
            conversation_id=conversation_id, role=role, content=content,
            resolved_question=resolved_question, sql=sql, chart_spec=chart_spec, status=status,
        ))
        session.commit()


def get_recent_turns(conversation_id: int, limit: int = 4) -> list[dict]:
    """Last few answered questions (oldest first), used to understand follow-up questions."""
    with SessionLocal() as session:
        rows = session.scalars(
            select(Message).where(Message.conversation_id == conversation_id)
            .order_by(Message.id.desc()).limit(limit * 2)
        ).all()
    rows = list(reversed(rows))
    turns, pending_question = [], None
    for message in rows:
        if message.role == "user":
            pending_question = message.content
        elif message.role == "assistant" and pending_question is not None:
            turns.append({
                "question": pending_question,
                "standalone_question": message.resolved_question or pending_question,
                "answer": message.content,
                "status": message.status,
            })
            pending_question = None
    return turns[-limit:]


def get_messages(conversation_id: int) -> list[dict]:
    with SessionLocal() as session:
        rows = session.scalars(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        ).all()
        return [
            {"role": m.role, "content": m.content, "sql": m.sql, "chart_spec": m.chart_spec,
             "status": m.status, "created_at": m.created_at.isoformat()}
            for m in rows
        ]


# --------------------------------------------------------------- query log

def log_query(dataset_id: int | None, question: str, sql: str | None, status: str,
              error: str | None, duration_ms: int) -> None:
    try:
        with SessionLocal() as session:
            session.add(QueryLog(
                dataset_id=dataset_id, question=question, sql=sql,
                status=status, error=error, duration_ms=duration_ms,
            ))
            session.commit()
    except Exception:
        logger.exception("Could not write query log")
