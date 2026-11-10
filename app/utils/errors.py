"""Custom exceptions. Each one carries a friendly message that is safe to show to the user."""


class InsightFlowError(Exception):
    """Base class. `user_message` is shown in the UI; technical details go to the log."""

    status_code = 400

    def __init__(self, user_message: str, technical_detail: str | None = None):
        super().__init__(technical_detail or user_message)
        self.user_message = user_message
        self.technical_detail = technical_detail


class DataLoadError(InsightFlowError):
    """The uploaded file could not be read or is not usable."""


class DatasetNotFoundError(InsightFlowError):
    status_code = 404


class SQLValidationError(InsightFlowError):
    """Generated SQL failed the safety checks."""


class SQLExecutionError(InsightFlowError):
    """Generated SQL was valid but the database could not run it."""


class LLMError(InsightFlowError):
    """The language model could not be reached or returned unusable output."""

    status_code = 502
