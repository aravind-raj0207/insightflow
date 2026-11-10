"""Reads an uploaded file into a Pandas DataFrame.

To support a new format later (for example JSON), write a reader function and
add it to READERS. Nothing else needs to change.
"""
import io
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from app.config import get_settings
from app.utils.errors import DataLoadError

logger = logging.getLogger(__name__)


def _read_csv(content: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            # sep=None lets pandas detect comma / semicolon / tab
            return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except pd.errors.EmptyDataError:
            raise DataLoadError("The CSV file is empty.")
        except pd.errors.ParserError as exc:
            raise DataLoadError("The CSV file could not be parsed. Please check its format.", str(exc))
    raise DataLoadError("The CSV file encoding is not supported.", str(last_error))


def _read_excel(content: bytes) -> pd.DataFrame:
    try:
        # First sheet only in version 1
        return pd.read_excel(io.BytesIO(content), sheet_name=0, engine="openpyxl")
    except Exception as exc:  # openpyxl raises many different error types
        raise DataLoadError("The Excel file could not be read. Is it a valid .xlsx file?", str(exc))


READERS: dict[str, Callable[[bytes], pd.DataFrame]] = {
    ".csv": _read_csv,
    ".xlsx": _read_excel,
    ".xlsm": _read_excel,
}


def load_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Validate the file and return its data as a DataFrame."""
    settings = get_settings()
    extension = Path(filename or "").suffix.lower()

    if extension not in READERS:
        supported = ", ".join(sorted(READERS))
        raise DataLoadError(f"Unsupported file type '{extension or 'unknown'}'. Please upload one of: {supported}")
    if not content:
        raise DataLoadError("The uploaded file is empty.")
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise DataLoadError(f"The file is larger than the {settings.max_upload_mb} MB limit.")

    df = READERS[extension](content)

    if df.shape[1] == 0 or df.shape[0] == 0:
        raise DataLoadError("The file has no data rows or no columns.")
    if df.shape[0] > settings.max_upload_rows:
        raise DataLoadError(f"The file has {df.shape[0]:,} rows; the limit is {settings.max_upload_rows:,}.")
    if df.shape[1] > settings.max_upload_columns:
        raise DataLoadError(f"The file has {df.shape[1]} columns; the limit is {settings.max_upload_columns}.")

    logger.info("Loaded %s: %d rows x %d columns", filename, df.shape[0], df.shape[1])
    return df
