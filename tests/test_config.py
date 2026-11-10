from app.config import Settings


def test_defaults_work_without_an_env_file():
    """A missing .env must not cause 'password authentication failed' with docker-compose defaults."""
    settings = Settings(_env_file=None)
    assert settings.db_password == "admin_dev_password"
    assert settings.db_readonly_password == "readonly_dev_password"
    assert settings.db_host == "127.0.0.1" and settings.db_port == 5433
    assert settings.admin_db_url.startswith("postgresql+psycopg2://insightflow_admin:admin_dev_password@127.0.0.1:5433/")
