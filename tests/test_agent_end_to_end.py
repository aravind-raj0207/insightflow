"""Full pipeline tests: upload -> agent -> SQL -> database -> chart -> answer.
Only the language model is replaced by a scripted fake (see fake_llm.py)."""
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.database.connection import readonly_engine
from tests.conftest import ask

pytestmark = pytest.mark.db
ZOMATO = pd.read_csv("sample_data/zomato_sample.csv")


def test_upload_profile_is_real(zomato):
    assert zomato["row_count"] == len(ZOMATO) and zomato["column_count"] == 6
    columns = {c["name"]: c for c in zomato["columns"]}
    assert columns["rating"]["kind"] == "numeric" and columns["rating"]["missing_count"] == int(ZOMATO.rating.isna().sum())
    assert columns["city"]["kind"] == "categorical"


def test_1_city_with_most_restaurants(client, zomato):
    result = ask(client, zomato, "Which city has the most restaurants?")
    assert result["status"] == "ok" and result["chart_type"] == "bar"
    assert result["rows"][0]["city"] == ZOMATO.city.value_counts().index[0]
    assert result["sql"].upper().startswith("SELECT") and result["figure"] is not None


def test_2_average_rating_by_city(client, zomato):
    result = ask(client, zomato, "What is the average rating by city?")
    expected = ZOMATO.groupby("city").rating.mean().round(2)
    for row in result["rows"]:
        assert row["avg_rating"] == pytest.approx(expected[row["city"]], abs=0.01)
    assert result["chart_type"] == "bar" and "bar" in result["chart_reason"].lower()


def test_3_monthly_sales_is_a_line_chart(client, sales):
    result = ask(client, sales, "Show monthly sales.")
    assert result["status"] == "ok" and result["chart_type"] == "line" and result["row_count"] == 12


def test_4_compare_chennai_and_bangalore(client, zomato):
    result = ask(client, zomato, "Compare Chennai and Bangalore.")
    assert {r["city"] for r in result["rows"]} == {"Chennai", "Bangalore"}
    assert any("higher than" in fact for fact in result["facts"])


def test_5_votes_vs_rating_is_a_scatter_plot(client, zomato):
    result = ask(client, zomato, "Show the relationship between votes and rating.")
    assert result["chart_type"] == "scatter"
    assert any("Correlation" in fact for fact in result["facts"])


def test_6_missing_information_is_not_invented(client, zomato):
    result = ask(client, zomato, "Which restaurant has the highest profit?")
    assert result["status"] == "cannot_answer"
    assert result["sql"] is None and result["figure"] is None
    assert "profit" in result["answer"] and "city" in result["answer"]    # honest + lists real columns


def test_7_explicit_chart_type_is_respected(client, sales):
    pie = ask(client, sales, "Show sales by region as a pie chart.")
    assert pie["chart_type"] == "pie"
    auto = ask(client, sales, "Show monthly sales.")
    bar = ask(client, sales, "Show monthly sales using a bar chart.")
    assert auto["chart_type"] == "line" and bar["chart_type"] == "bar"


def test_8a_request_to_delete_data_is_refused(client, zomato):
    result = ask(client, zomato, "Delete all rows from the table")
    assert result["status"] == "rejected" and result["sql"] is None
    assert ask(client, zomato, "Which city has the most restaurants?")["status"] == "ok"   # data intact


def test_8b_destructive_sql_from_the_model_is_blocked(client, zomato):
    result = ask(client, zomato, "tricky question")                  # fake model answers with DROP TABLE
    assert result["status"] == "error" and "DROP" not in result["answer"]
    still_there = ask(client, zomato, "Which city has the most restaurants?")
    assert still_there["status"] == "ok" and sum(r["restaurant_count"] for r in still_there["rows"]) == len(ZOMATO)


def test_8c_database_user_cannot_write_even_if_validation_failed(zomato):
    with pytest.raises(DBAPIError):
        with readonly_engine.begin() as conn:
            conn.execute(text(f'DELETE FROM user_data."{zomato["table_name"]}"'))
    with pytest.raises(DBAPIError):
        with readonly_engine.begin() as conn:
            conn.execute(text("SELECT * FROM app.datasets"))


def test_follow_up_questions(client, zomato):
    first = ask(client, zomato, "Which city has the highest average rating?")
    top_city = first["rows"][0]["city"]
    assert top_city in first["answer"] or top_city in first["facts"][0]

    second = ask(client, zomato, "How many restaurants are there?", first["conversation_id"])
    assert top_city in second["resolved_question"]                     # "there" -> the top city
    assert second["rows"][0]["restaurant_count"] == int((ZOMATO.city == top_city).sum())

    third = ask(client, zomato, "What about Chennai?", first["conversation_id"])
    assert "Chennai" in third["resolved_question"]
    assert third["rows"][0]["restaurant_count"] == int((ZOMATO.city == "Chennai").sum())


def test_unrelated_question_is_not_confused_by_history(client, zomato):
    first = ask(client, zomato, "Which city has the highest average rating?")
    unrelated = ask(client, zomato, "What is the average rating by city?", first["conversation_id"])
    assert unrelated["resolved_question"] == "What is the average rating by city?"


def test_self_correction_after_hallucinated_column(client, zomato, fake_llm):
    before = fake_llm.calls.count("generate_sql")
    result = ask(client, zomato, "wrong column first")
    assert result["status"] == "ok" and fake_llm.calls.count("generate_sql") - before == 2


def test_empty_result_is_reported_honestly(client, zomato):
    result = ask(client, zomato, "empty result")
    assert result["status"] == "empty" and result["figure"] is None


def test_conversation_cannot_cross_datasets(client, zomato, sales):
    conversation = ask(client, zomato, "Which city has the most restaurants?")["conversation_id"]
    response = client.post("/api/chat", json={"dataset_id": sales["id"], "question": "Show monthly sales",
                                              "conversation_id": conversation})
    assert response.status_code == 404


def test_each_dataset_has_its_own_table(zomato, sales):
    assert zomato["table_name"] != sales["table_name"]


def test_upload_errors_are_friendly(client):
    for name, content in [("notes.txt", b"hello"), ("empty.csv", b""), ("headers_only.csv", b"a,b\n")]:
        response = client.post("/api/datasets/upload", files={"file": (name, content)})
        assert response.status_code == 400
        assert "Traceback" not in response.text and response.json()["detail"]


def test_unknown_dataset_gives_404(client):
    assert client.post("/api/chat", json={"dataset_id": 999999, "question": "hi"}).status_code == 404


def test_model_that_invents_a_column_gets_an_honest_answer(client, zomato):
    result = ask(client, zomato, "invented column")
    assert result["status"] == "cannot_answer" and "profit" in result["answer"]
    assert result["figure"] is None


def test_percent_signs_in_sql_work(client, zomato):
    result = ask(client, zomato, "name pattern")
    assert result["status"] == "ok" and result["rows"][0]["n"] == 99      # Restaurant 0001 .. 0099


def test_database_setup_repairs_itself(client):
    from app.database.connection import check_database, readonly_engine
    from app.database.repository import init_db
    init_db()
    init_db()                                      # safe to run repeatedly
    assert check_database(readonly_engine)
