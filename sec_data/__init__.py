"""SEC EDGAR download, cleaning, and amendment-audit interfaces."""

from sec_data.apple_case import AppleAmendmentCase, load_apple_amendment_case
from sec_data.client import SecConfigurationError, SecRequestError
from sec_data.models import SecIngestionResult
from sec_data.pipeline import SecIngestionService

__all__ = [
    "AppleAmendmentCase",
    "SecConfigurationError",
    "SecIngestionResult",
    "SecIngestionService",
    "SecRequestError",
    "load_apple_amendment_case",
]
