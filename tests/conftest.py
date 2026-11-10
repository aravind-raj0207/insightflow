import pytest
from fastapi.testclient import TestClient

from app.agent.runner import AnalystService
from app.api.routes_chat import get_analyst_service
from app.database.connection import admin_engine, check_database
from tests.fake_llm import FakeLLM


def pytest_collection_modifyitems(config, items):
    """Skip database tests automatically when PostgreSQL is not running."""
    if check_database(admin_engine):
        return
    skip = pytest.mark.skip(reason="PostgreSQL is not running (docker compose up -d)")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture(scope="session")
def client(fake_llm):
    from app.main import app
    from app.database.repository import init_db
    init_db()
    service = AnalystService(fake_llm)
    app.dependency_overrides[get_analyst_service] = lambda: service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _upload(client, path, content_type):
    with open(path, "rb") as handle:
        response = client.post("/api/datasets/upload", files={"file": (path.split("/")[-1], handle, content_type)})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="session")
def zomato(client):
    dataset = _upload(client, "sample_data/zomato_sample.csv", "text/csv")
    yield dataset
    client.delete(f"/api/datasets/{dataset['id']}")


@pytest.fixture(scope="session")
def sales(client):
    dataset = _upload(client, "sample_data/sales_sample.xlsx",
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    yield dataset
    client.delete(f"/api/datasets/{dataset['id']}")


def ask(client, dataset, question, conversation_id=None):
    response = client.post("/api/chat", json={"dataset_id": dataset["id"], "question": question,
                                              "conversation_id": conversation_id})
    assert response.status_code == 200, response.text
    return response.json()
