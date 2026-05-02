"""
logger.py
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a logger with console and optional file handlers.

    Args:
        name:     Logger name
        log_file: Optional path to log file
        level:    Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class MetricTracker:
    """Tracks running averages of scalar metrics during an epoch."""

    def __init__(self):
        self._values = {}
        self._counts = {}

    def update(self, key: str, value: float, n: int = 1) -> None:
        if key not in self._values:
            self._values[key] = 0.0
            self._counts[key] = 0
        self._values[key] += value * n
        self._counts[key] += n

    def avg(self, key: str) -> float:
        if self._counts.get(key, 0) == 0:
            return 0.0
        return self._values[key] / self._counts[key]

    def reset(self) -> None:
        self._values.clear()
        self._counts.clear()

    def __str__(self) -> str:
        parts = [f"{k}={self.avg(k):.4f}" for k in self._values]
        return " | ".join(parts)
