"""All settings live here. Values come from the .env file / environment variables."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# The .env file is looked up next to the app folder, no matter where you start the server from.
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    gemini_api_version: str = "v1beta"        # the API version that supports model aliases like gemini-flash-latest
    gemini_temperature: float | None = None   # None = use the model default (recommended)

    db_host: str = "127.0.0.1"   # 127.0.0.1 (IPv4) avoids Windows "localhost -> ::1" surprises
    db_port: int = 5433         # docker-compose publishes PostgreSQL on 5433 to avoid clashing with a local install
    db_name: str = "insightflow"
    db_user: str = "insightflow_admin"
    db_password: str = "admin_dev_password"
    db_readonly_user: str = "insightflow_readonly"
    db_readonly_password: str = "readonly_dev_password"

    max_upload_mb: int = 50
    max_upload_rows: int = 500_000
    max_upload_columns: int = 200
    query_timeout_seconds: int = 10
    max_result_rows: int = 1000

    api_base_url: str = "http://127.0.0.1:8000"

    model_config = SettingsConfigDict(env_file=str(ENV_FILE), extra="ignore")

    @property
    def admin_db_url(self) -> str:
        return (f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}")

    @property
    def readonly_db_url(self) -> str:
        return (f"postgresql+psycopg2://{self.db_readonly_user}:{self.db_readonly_password}"
                f"@{self.db_host}:{self.db_port}/{self.db_name}")


@lru_cache
def get_settings() -> Settings:
    """Create the Settings object once and reuse it everywhere."""
    return Settings()
