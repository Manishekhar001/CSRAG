"""Logging configuration — identical pattern to BasicRAG project."""

import logging
import sys
from functools import lru_cache


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logger for the application.

    Args:
        log_level: One of DEBUG, INFO, WARNING, ERROR, CRITICAL.
    """
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicate logs on reload
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # Silence noisy third-party libraries
    for noisy_lib in (
        "httpx",
        "httpcore",
        "openai",
        "qdrant_client",
        "urllib3",
        "groq",
        "ollama",
        "langgraph",
    ):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


@lru_cache
def get_logger(name: str) -> logging.Logger:
    """Return a cached logger for the given module name.

    Args:
        name: Typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin that adds a ``logger`` property to any class."""

    @property
    def logger(self) -> logging.Logger:
        """Return logger bound to this class name."""
        return get_logger(self.__class__.__name__)
