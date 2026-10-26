import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str = "ege_ai_agent", log_file: str | None = None) -> logging.Logger:
    """Configures and returns a logger supporting console and rotating file handlers across all logging levels."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    raw_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, raw_level, logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler (logs/scraibe.log by default, configurable via LOG_TO_FILE and LOG_FILE)
    log_to_file = os.getenv("LOG_TO_FILE", "1").strip() in {"1", "true", "True"}
    if log_to_file or log_file:
        file_path_str = log_file or os.getenv("LOG_FILE", "logs/scraibe.log")
        try:
            file_path = Path(file_path_str)
            if not file_path.is_absolute():
                workspace_root = Path(__file__).resolve().parents[2]
                file_path = workspace_root / file_path_str
            file_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                filename=str(file_path),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(level)
            logger.addHandler(file_handler)
        except Exception as exc:
            # Fallback warning if file logging cannot be set up (e.g. read-only fs)
            console_handler.handle(
                logger.makeRecord(
                    name=name,
                    level=logging.WARNING,
                    fn="",
                    lno=0,
                    msg=f"Could not initialize file log handler at {file_path_str}: {exc}",
                    args=(),
                    exc_info=None,
                )
            )

    return logger
