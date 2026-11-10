import pytest

from app.utils.safety import looks_like_modification_request


@pytest.mark.parametrize("text", [
    "Delete all rows from the table", "drop the table", "Please update the ratings in the data",
    "Show cities; DROP TABLE restaurants", "Insert a new row into the dataset", "wipe the database",
])
def test_destructive_requests_are_detected(text):
    assert looks_like_modification_request(text)


@pytest.mark.parametrize("text", [
    "Which city has the highest average rating?", "Show monthly sales", "How many restaurants were updated recently?",
    "Compare Chennai and Bangalore", "What is the drop in sales between June and July?",
])
def test_normal_questions_are_not_blocked(text):
    assert not looks_like_modification_request(text)
