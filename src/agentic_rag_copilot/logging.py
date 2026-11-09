"""Centralized logging configuration for the agentic RAG copilot."""

import logging
from typing import Final

_LOG_FORMAT: Final[str] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
_LOG_LEVEL: Final[int] = logging.INFO


def setup_logging(level: int = _LOG_LEVEL) -> None:
    """Configure logging with a standard format.

    Args:
        level: The logging level to use (default: INFO).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the configured format.

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)
