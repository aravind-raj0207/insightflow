import logging
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from app.data.cleaner import clean_dataframe
from app.data.loader import load_dataframe
from app.data.profiler import profile_dataframe
from app.database import repository
from app.utils.errors import DataLoadError

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    """Upload a CSV / Excel file: read -> clean -> profile -> store in PostgreSQL."""
    filename = Path(file.filename or "").name
    content = await file.read()

    raw_df = load_dataframe(filename, content)
    df, report = clean_dataframe(raw_df)
    profile = profile_dataframe(df)

    # remember the original header text next to each cleaned column name
    renamed = {r["new"]: r["original"] for r in report.renamed_columns}
    for column in profile["columns"]:
        column["original_name"] = renamed.get(column["name"], column["name"])
    profile["cleaning"] = report.to_dict()

    if profile["column_count"] == 0:
        raise DataLoadError("No usable columns were found in the file.")

    dataset = repository.create_dataset(df, filename, profile)
    logger.info("Dataset %s stored as %s", dataset["id"], dataset["table_name"])
    return dataset
