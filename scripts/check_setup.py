"""Checks that everything InsightFlow needs is working, and tells you how to fix what is not.

Run from the project folder (with the virtual environment active):
    python scripts/check_setup.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from app.config import ENV_FILE, get_settings  # noqa: E402

settings = get_settings()
problems = 0


def ok(message: str) -> None:
    print(f"  [ OK ]  {message}")


def fail(message: str, fix: str) -> None:
    global problems
    problems += 1
    print(f"  [FAIL]  {message}\n          FIX: {fix}")


def warn(message: str) -> None:
    print(f"  [WARN]  {message}")


print("\nInsightFlow setup check\n")

# 1. .env file
if ENV_FILE.exists():
    ok(f".env file found ({ENV_FILE})")
else:
    warn(f".env file not found at {ENV_FILE}. Built-in defaults are used. "
         "Create it with:  copy .env.example .env   (Windows)  /  cp .env.example .env  (Mac/Linux)")

# 2. database
print(f"\nDatabase: {settings.db_host}:{settings.db_port}  name={settings.db_name}  user={settings.db_user}")
from app.database.connection import admin_engine, describe_db_error, readonly_engine  # noqa: E402

reason = describe_db_error(admin_engine)
if reason is None:
    ok("connected to PostgreSQL as the admin user")
    from app.database.repository import init_db  # noqa: E402
    try:
        init_db()
        ok("schemas, tables and the read-only user are ready")
    except Exception as exc:
        fail(f"could not prepare the database: {exc}", "run: docker compose down -v   then   docker compose up -d")

    reason = describe_db_error(readonly_engine)
    if reason is None:
        ok("connected as the read-only user")
        try:
            with readonly_engine.begin() as conn:
                conn.execute(text("CREATE TABLE user_data.__should_fail (a int)"))
            fail("the read-only user was able to CREATE a table!", "run: docker compose down -v   then   docker compose up -d")
        except Exception:
            ok("the read-only user cannot write (this is what we want)")
    else:
        fail(f"read-only user cannot connect: {reason}", "run: docker compose down -v   then   docker compose up -d")
else:
    low = reason.lower()
    if "password authentication failed" in low:
        fix = (f"Another PostgreSQL may be using port {settings.db_port}, or an old Docker volume has other credentials. "
               "Run: docker compose down -v   then   docker compose up -d   (this resets the database). "
               "Make sure DB_PORT=5433 in .env.")
    elif "refused" in low or "could not connect" in low or "timeout" in low:
        fix = "PostgreSQL is not running. Start Docker Desktop, then run: docker compose up -d"
    elif "does not exist" in low:
        fix = "Run: docker compose down -v   then   docker compose up -d"
    else:
        fix = "Check the DB_* values in .env and that Docker is running (docker compose up -d)"
    fail(f"cannot connect to PostgreSQL: {reason}", fix)

# 3. Gemini
print(f"\nGemini model: {settings.gemini_model}")
if not settings.gemini_api_key or settings.gemini_api_key == "your_key_here":
    fail("GEMINI_API_KEY is not set", "get a key at https://aistudio.google.com/apikey and put it in .env")
else:
    try:
        from app.llm.gemini_client import GeminiClient  # noqa: E402
        client = GeminiClient(settings.gemini_api_key, settings.gemini_model, settings.gemini_temperature,
                              settings.gemini_api_version or None)
        reply = client.generate("Reply with the single word OK.", task="check_setup").strip()
        ok(f"Gemini answered: {reply[:40]!r} (model used: {client.model})")
        if client.model != settings.gemini_model:
            warn(f"'{settings.gemini_model}' was not available, so '{client.model}' was used. "
                 f"To make it permanent, set GEMINI_MODEL={client.model} in .env")
    except Exception as exc:
        detail = getattr(exc, "technical_detail", None) or str(exc)
        message = getattr(exc, "user_message", "") or ""
        if "busy" in message.lower() or "503" in detail or "UNAVAILABLE" in detail:
            fix = ("Google's model is overloaded right now. Your key and model are fine. Wait a minute and run this "
                   "check again. The app also switches to another Flash model by itself when this happens.")
        elif "not available for your API key" in message or "NOT_FOUND" in detail:
            fix = "run python scripts/list_models.py and put one of the listed names in GEMINI_MODEL in .env"
        else:
            fix = "check the key in .env (https://aistudio.google.com/apikey) and your internet connection"
        fail(f"Gemini call failed: {detail[:300]}", fix)

print("\n" + ("Everything looks good. Start the app:\n  uvicorn app.main:app --reload\n  streamlit run frontend/streamlit_app.py\n"
              if problems == 0 else f"{problems} problem(s) found. Fix them and run this check again.\n"))
sys.exit(1 if problems else 0)
