import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import routes_chat, routes_dataset, routes_upload
from sqlalchemy.exc import OperationalError

from app.config import ENV_FILE, get_settings
from app.database.connection import admin_engine, check_database, describe_db_error, readonly_engine
from app.database.repository import init_db
from app.utils.errors import InsightFlowError
from app.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("insightflow")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not ENV_FILE.exists():
        logger.warning("No .env file found at %s - using built-in development defaults. "
                       "Copy .env.example to .env and add your GEMINI_API_KEY.", ENV_FILE)
    try:
        init_db()
    except Exception:
        reason = describe_db_error(admin_engine) or "unknown error"
        logger.error(
            "Could not connect to PostgreSQL at %s:%s as '%s': %s\n"
            "  -> Is Docker running?  Try: docker compose up -d\n"
            "  -> Run: python scripts/check_setup.py   to diagnose the problem.",
            settings.db_host, settings.db_port, settings.db_user, reason)
        logger.debug("Full database error", exc_info=True)
    yield


app = FastAPI(title="InsightFlow", description="AI-Powered Natural Language Data Analytics", lifespan=lifespan)
app.include_router(routes_upload.router)
app.include_router(routes_dataset.router)
app.include_router(routes_chat.router)


@app.exception_handler(InsightFlowError)
async def handle_known_error(request: Request, exc: InsightFlowError):
    """Known problems: show a friendly message, keep the technical detail in the log only."""
    logger.warning("%s: %s", type(exc).__name__, exc.technical_detail or exc.user_message)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.user_message})


@app.exception_handler(OperationalError)
async def handle_database_down(request: Request, exc: OperationalError):
    logger.error("Database connection problem: %s", str(getattr(exc, "orig", exc)).splitlines()[0])
    return JSONResponse(status_code=503, content={
        "detail": "The database is not reachable. Make sure Docker is running (docker compose up -d) "
                  "and that the DB_* settings in .env are correct."})


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Something went wrong on the server. Please try again."})


@app.get("/")
def root() -> dict:
    return {"app": "InsightFlow", "status": "running"}


@app.get("/health")
def health() -> dict:
    admin_ok, readonly_ok = check_database(admin_engine), check_database(readonly_engine)
    result = {"status": "ok" if admin_ok and readonly_ok else "degraded",
              "admin_db": admin_ok, "readonly_db": readonly_ok}
    if not (admin_ok and readonly_ok):
        result["hint"] = describe_db_error(admin_engine if not admin_ok else readonly_engine)
    return result
