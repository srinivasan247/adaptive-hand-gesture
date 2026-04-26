"""
Logging configuration.
"""

import logging
import logging.handlers
from config.settings import LOG_DIR


def setup_logger(level=logging.INFO):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "gesture_control.log"

    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler (rotating)
    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)
