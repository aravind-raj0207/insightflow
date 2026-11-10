"""SQL validation. This is deterministic code - the LLM has no say here.

We PARSE the SQL into a syntax tree (with sqlglot) instead of searching for bad
words, so tricks like comments or odd spacing cannot sneak past.

Rules:
  1. exactly one statement
  2. it must be a SELECT query (WITH ... SELECT and UNION are fine)
  3. no write/DDL/locking nodes anywhere inside it
  4. it may only read the active dataset's table (or its own CTEs)
  5. no dangerous functions (pg_sleep, pg_read_file, dblink ...)
  6. every column must exist in the dataset (or be an alias defined in the query)
  7. a row LIMIT is always enforced
"""
import logging
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

logger = logging.getLogger(__name__)

# Looked up by name so the code keeps working across sqlglot versions.
_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Drop", "Create", "Alter", "AlterTable", "Command",
    "TruncateTable", "Merge", "Copy", "Set", "Use", "Into", "Lock", "Grant", "Revoke",
    "Transaction", "Commit", "Rollback", "LoadData", "Attach", "Detach",
]
FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))

BLOCKED_FUNCTIONS = {
    "dblink", "lo_import", "lo_export", "set_config", "current_setting", "version",
    "generate_series", "txid_current", "query_to_xml", "xpath", "unnest",
}


@dataclass
class ValidationResult:
    is_valid: bool
    sql: str = ""          # cleaned SQL that is safe to run (only when is_valid)
    pretty_sql: str = ""   # nicely formatted version for display
    error: str = ""        # explanation when not valid


def _fail(message: str) -> ValidationResult:
    return ValidationResult(is_valid=False, error=message)


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def _enforce_limit(tree: exp.Expression, max_rows: int) -> exp.Expression:
    limit_node = tree.args.get("limit")
    if limit_node is not None:
        try:
            current = int(limit_node.expression.name)
            if current <= max_rows:
                return tree
        except (ValueError, AttributeError):
            pass  # LIMIT with something odd (e.g. a function) -> replace it
    return tree.limit(max_rows)


def validate_sql(sql: str, allowed_table: str, allowed_columns: list[str], max_rows: int) -> ValidationResult:
    if not sql or not sql.strip():
        return _fail("No SQL was generated.")

    # Rule 1: a single statement
    try:
        statements = [s for s in sqlglot.parse(sql, read="postgres") if s is not None]
    except SqlglotError as exc:
        return _fail(f"The SQL has a syntax error: {str(exc)[:200]}")
    if len(statements) != 1:
        return _fail("Only a single SQL statement is allowed.")
    tree = statements[0]

    # Rule 2: must be a SELECT-style query
    if not isinstance(tree, exp.Query):
        return _fail("Only SELECT queries are allowed.")

    # Rule 3: nothing that writes, changes structure or locks rows
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return _fail(f"The query contains a forbidden operation ({type(node).__name__.upper()}).")

    # Rule 4: only the active dataset's table
    cte_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
    allowed = allowed_table.lower()
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        schema = table.db.lower() if table.db else ""
        if table.catalog:
            return _fail("Cross-database access is not allowed.")
        if name in cte_names and not schema:
            continue
        if name == allowed and schema in ("", "user_data"):
            continue
        return _fail(f"Access to '{table.sql()}' is not allowed. Only the table '{allowed_table}' can be queried.")

    # Rule 5: no dangerous functions
    for func in tree.find_all(exp.Func):
        name = _function_name(func)
        if name.startswith("pg_") or name in BLOCKED_FUNCTIONS:
            return _fail(f"The function '{name}' is not allowed.")

    # Rule 6: columns must exist (this catches hallucinated column names before the database does)
    defined_names = {c.lower() for c in allowed_columns}
    defined_names |= {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    defined_names |= {c for c in cte_names}
    for table_alias in tree.find_all(exp.TableAlias):
        defined_names |= {col.name.lower() for col in table_alias.columns}
    for column in tree.find_all(exp.Column):
        if column.name and column.name.lower() not in defined_names:
            return _fail(
                f"The column '{column.name}' does not exist. Available columns: {', '.join(allowed_columns)}"
            )

    # Rule 7: always limit the number of rows returned
    tree = _enforce_limit(tree, max_rows)

    return ValidationResult(
        is_valid=True,
        sql=tree.sql(dialect="postgres"),
        pretty_sql=tree.sql(dialect="postgres", pretty=True),
    )
