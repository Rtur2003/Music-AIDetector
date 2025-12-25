"""
Centralized logging configuration for the Music AI Detector.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from .config import get_config
except ImportError:
    from config import get_config


class LoggerSetup:
    """Centralized logger setup with consistent formatting."""

    _initialized = False

    @classmethod
    def setup_logger(
        cls,
        name: str,
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
    ) -> logging.Logger:
        """
        Create or get a logger with consistent formatting.

        Args:
            name: Logger name (typically __name__)
            level: Logging level (default: INFO)
            log_file: Optional file path for logging

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)

        if not cls._initialized or not logger.handlers:
            logger.setLevel(level)

            # Console handler with detailed format
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Optional file handler
            if log_file:
                log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

            cls._initialized = True

        return logger


def get_logger(name: str, log_to_file: bool = False) -> logging.Logger:
    """
    Convenience function to get a logger.

    Args:
        name: Logger name
        log_to_file: Whether to log to file in addition to console

    Returns:
        Configured logger instance
    """
    log_file = None
    if log_to_file:
        try:
            cfg = get_config()
            log_file = cfg.data_dir / "logs" / f"{name.split('.')[-1]}.log"
        except Exception:
            pass

    return LoggerSetup.setup_logger(name, log_file=log_file)
