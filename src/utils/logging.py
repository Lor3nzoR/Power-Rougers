"""
Logging centralizzato. Usa Rich se disponibile, altrimenti fallback stdlib.
"""
from __future__ import annotations
import logging
import sys

try:
    from rich.logging import RichHandler
    _RICH = True
except ImportError:
    _RICH = False


def get_logger(name: str = "mirror", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    if _RICH:
        handler = RichHandler(rich_tracebacks=True, show_time=True, show_path=False)
        fmt = "%(message)s"
    else:
        handler = logging.StreamHandler(sys.stderr)
        fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


log = get_logger()
