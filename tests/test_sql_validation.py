import pytest

from app.agent.nodes.sql_validation import validate_sql

TABLE, COLUMNS = "ds_abc123", ["city", "rating", "votes"]


def check(sql):
    return validate_sql(sql, TABLE, COLUMNS, max_rows=1000)


def test_valid_select_passes_and_gets_a_limit():
    result = check(f"SELECT city, AVG(rating) AS avg_rating FROM user_data.{TABLE} GROUP BY city")
    assert result.is_valid and "LIMIT 1000" in result.sql


def test_cte_and_union_are_allowed():
    assert check(f"WITH t AS (SELECT city, COUNT(*) AS n FROM {TABLE} GROUP BY city) SELECT city, n FROM t").is_valid


def test_large_limit_is_reduced():
    assert "LIMIT 1000" in check(f"SELECT city FROM {TABLE} LIMIT 99999").sql


@pytest.mark.parametrize("sql", [
    f"DELETE FROM {TABLE}",
    f"UPDATE {TABLE} SET rating = 1",
    f"INSERT INTO {TABLE} (city) VALUES ('x')",
    f"DROP TABLE {TABLE}",
    f"ALTER TABLE {TABLE} ADD COLUMN x int",
    f"TRUNCATE TABLE {TABLE}",
    "CREATE TABLE evil (a int)",
    f"SELECT * INTO copy FROM {TABLE}",
    f"SELECT city FROM {TABLE} FOR UPDATE",
])
def test_write_and_ddl_statements_are_rejected(sql):
    assert not check(sql).is_valid


def test_multiple_statements_are_rejected():
    assert not check(f"SELECT city FROM {TABLE}; DROP TABLE {TABLE}").is_valid


@pytest.mark.parametrize("sql", [
    "SELECT * FROM app.datasets",
    "SELECT * FROM information_schema.tables",
    "SELECT usename FROM pg_user",
    f"SELECT city FROM {TABLE} UNION SELECT table_name FROM information_schema.tables",
    "SELECT * FROM user_data.ds_someone_elses_table",       # another dataset
])
def test_other_tables_are_rejected(sql):
    assert not check(sql).is_valid


@pytest.mark.parametrize("sql", [
    f"SELECT pg_sleep(30) FROM {TABLE}",
    f"SELECT pg_read_file('/etc/passwd') FROM {TABLE}",
    f"SELECT set_config('a','b',false) FROM {TABLE}",
])
def test_dangerous_functions_are_rejected(sql):
    assert not check(sql).is_valid


def test_hallucinated_column_is_rejected_with_helpful_message():
    result = check(f"SELECT profit FROM {TABLE}")
    assert not result.is_valid and "profit" in result.error and "city" in result.error


def test_syntax_error_and_empty_sql_are_rejected():
    assert not check("SELEC city FRM").is_valid
    assert not check("").is_valid


def test_comment_cannot_hide_a_second_statement():
    assert not check(f"SELECT city FROM {TABLE}; -- harmless\nDELETE FROM {TABLE}").is_valid


@pytest.mark.parametrize("sql", [
    f"SELECT city, COUNT(*) AS n, 100.0 * COUNT(*) / NULLIF(SUM(COUNT(*)) OVER (), 0) AS pct FROM {TABLE} GROUP BY city ORDER BY pct DESC",
    f"SELECT city, CASE WHEN rating >= 4 THEN 'high' ELSE 'low' END AS band, COUNT(*) AS n FROM {TABLE} GROUP BY 1, 2",
    f"SELECT city FROM {TABLE} WHERE city ILIKE '%ban%' AND rating BETWEEN 3 AND 5 ORDER BY rating DESC NULLS LAST",
    f"SELECT DATE_TRUNC('month', rating) AS month, SUM(votes) AS total FROM {TABLE} GROUP BY 1 ORDER BY 1",
    f"SELECT EXTRACT(YEAR FROM votes) AS y, COUNT(*) FROM {TABLE} GROUP BY 1",
    f"SELECT t.city, t.avg_r FROM (SELECT city, AVG(rating) AS avg_r FROM {TABLE} GROUP BY city) AS t WHERE t.avg_r > 4",
    f"SELECT city, AVG(rating)::numeric(10,2) AS a, COALESCE(MAX(votes), 0) AS m FROM {TABLE} GROUP BY city HAVING COUNT(*) > 5",
    f"SELECT DISTINCT city FROM user_data.{TABLE} AS r ORDER BY r.city",
    f"SELECT city, ROUND(AVG(rating)::numeric, 2) AS \"Average Rating\" FROM {TABLE} GROUP BY city ORDER BY \"Average Rating\" DESC",
])
def test_realistic_analytical_queries_are_accepted(sql):
    result = check(sql)
    assert result.is_valid, result.error
