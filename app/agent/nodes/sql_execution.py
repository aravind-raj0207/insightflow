"""Runs validated SQL with the READ-ONLY database user (deterministic)."""
import datetime
import logging
from decimal import Decimal

import pandas as pd
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from app.agent.nodes.sql_validation import validate_sql
from app.config import get_settings
from app.database.connection import readonly_engine

logger = logging.getLogger(__name__)


def normalize_result_types(df: pd.DataFrame) -> pd.DataFrame:
    """PostgreSQL NUMERIC arrives as Python Decimal and DATE as datetime.date, which Pandas treats as
    plain objects (so charts would think they are categories). Convert them to float / datetime."""
    for column in df.columns:
        if df[column].dtype != object:
            continue
        values = df[column].dropna()
        if values.empty:
            continue
        if all(isinstance(v, Decimal) for v in values):
            df[column] = pd.to_numeric(df[column].astype(float))
        elif all(isinstance(v, datetime.date) for v in values):
            df[column] = pd.to_datetime(df[column])
    return df


def validate_sql_node(state: dict) -> dict:
    settings = get_settings()
    result = validate_sql(
        state["sql"], allowed_table=state["table_name"],
        allowed_columns=[c["name"] for c in state["columns"]], max_rows=settings.max_result_rows,
    )
    if not result.is_valid:
        logger.warning("SQL rejected: %s | %s", result.error, state["sql"])
        return {"sql_error": f"Validation: {result.error}"}
    return {"sql": result.sql, "pretty_sql": result.pretty_sql, "sql_error": None}


def execute_sql_node(state: dict, engine=None) -> dict:
    settings = get_settings()
    engine = engine or readonly_engine
    timeout_ms = int(settings.query_timeout_seconds) * 1000
    try:
        with engine.connect() as conn:
            with conn.begin():
                # Belt and braces: read-only transaction + timeout + only the user_data schema visible
                conn.exec_driver_sql("SET TRANSACTION READ ONLY")
                conn.exec_driver_sql(f"SET LOCAL statement_timeout = {timeout_ms}")
                conn.exec_driver_sql("SET LOCAL search_path TO user_data")
                # '%' must be doubled because the driver treats it as a parameter marker
                cursor = conn.exec_driver_sql(state["sql"].replace("%", "%%"))
                df = normalize_result_types(pd.DataFrame(cursor.fetchall(), columns=list(cursor.keys())))
        return {"result_df": df, "sql_error": None}
    except DBAPIError as exc:
        message = str(getattr(exc, "orig", exc)).strip().splitlines()[0]
        logger.warning("SQL execution failed: %s | %s", message, state["sql"])
        return {"sql_error": f"Execution: {message}"}
    except SQLAlchemyError as exc:
        logger.exception("Unexpected database error")
        return {"sql_error": f"Execution: {exc}"}
