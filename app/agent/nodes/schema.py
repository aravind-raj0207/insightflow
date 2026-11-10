"""Schema node (deterministic): loads the real column information from the database
and builds the text the LLM receives. The LLM never has to guess what columns exist."""
from app.database import repository
from app.utils.formatting import fmt_number


def build_schema_text(dataset: dict) -> str:
    lines = [
        f"Table: user_data.{dataset['table_name']}   (from file '{dataset['name']}', {dataset['row_count']:,} rows)",
        "Columns:",
    ]
    for col in dataset["columns"]:
        line = f"- {col['name']} ({col['sql_type']}, {col['kind']}"
        if col["missing_count"]:
            line += f", {col['missing_count']:,} missing"
        line += ")"
        stats = col.get("stats") or {}
        if col["kind"] == "numeric" and stats:
            line += f" range {fmt_number(stats.get('min'))} to {fmt_number(stats.get('max'))}"
        elif col["kind"] == "datetime" and stats:
            line += f" from {str(stats.get('min'))[:10]} to {str(stats.get('max'))[:10]}"
        elif col["top_values"]:
            values = ", ".join(f"'{v['value']}'" for v in col["top_values"][:15])
            line += f" Top values: {values}"
        lines.append(line)
    return "\n".join(lines)


def load_schema_node(state: dict) -> dict:
    dataset = repository.get_dataset(state["dataset_id"])
    return {
        "table_name": dataset["table_name"],
        "columns": dataset["columns"],
        "schema_text": build_schema_text(dataset),
        "attempts": 0,
        "sql_error": None,
    }
