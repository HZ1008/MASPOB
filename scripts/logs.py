"""Simple colored logger with file output for MASPOB.

This module provides a lightweight logging utility that prints colored
messages to the console and writes them to a log file under the
``MASPOB`` project directory.
"""

import os
from datetime import datetime
from enum import Enum
from typing import Optional, TextIO, Union


class Colors:
    """Terminal color codes for different log levels."""

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class LogLevel(Enum):
    """Log levels with corresponding colors."""

    DEBUG = (10, Colors.BLUE)
    INFO = (20, Colors.GREEN)
    WARNING = (30, Colors.YELLOW)
    ERROR = (40, Colors.RED)
    CRITICAL = (50, Colors.MAGENTA)


class SimpleLogger:
    """Simple logger with colored console output and file logging.

    By default, log files are stored under ``<project_root>/logs`` regardless of
    the current working directory. This is implemented by resolving the path
    relative to this file (``scripts/logs.py``).
    """

    def __init__(
        self,
        name: str = "MASPOB",
        log_level: Union[int, LogLevel] = LogLevel.INFO,
        log_file: Optional[str] = None,
        log_dir: Optional[str] = None,
        console_output: bool = True,
    ) -> None:
        """Initialize the logger.

        Args:
            name: Logger name (used in default filename).
            log_level: Minimum log level to display.
            log_file: Log file name. If ``None``, use ``{name}_YYYY-MM-DD.log``.
            log_dir: Directory to store log files. If ``None``, logs are stored
                under ``<project_root>/logs`` (one level above ``scripts/``).
            console_output: Whether to output logs to console.
        """

        self.name = name

        # Normalize log level
        if isinstance(log_level, LogLevel):
            self.log_level = log_level.value[0]
        else:
            self.log_level = int(log_level)

        self.console_output = console_output
        self.file_output: Optional[TextIO] = None

        # Resolve default log directory: <project_root>/logs
        if log_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            log_dir = os.path.join(base_dir, "logs")

        self.log_dir = log_dir

        # Set up file logging if a directory is provided
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

            if log_file is None:
                current_date = datetime.now().strftime("%Y-%m-%d")
                log_file = f"{self.name}_{current_date}.log"

            file_path = os.path.join(self.log_dir, log_file)
            self.file_output = open(file_path, "a", encoding="utf-8")

    # ------------------------------------------------------------------
    # Core logging implementation
    # ------------------------------------------------------------------
    def _log(self, level: LogLevel, message: str) -> None:
        """Internal method to log messages at the specified level."""

        if level.value[0] < self.log_level:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level_name = level.name
        formatted_msg = f"{timestamp} - {level_name} - {message}"

        # Console output with colors
        if self.console_output:
            color = level.value[1]
            if level == LogLevel.CRITICAL:
                colored_msg = f"{Colors.BOLD}{color}{formatted_msg}{Colors.RESET}"
            else:
                colored_msg = f"{color}{formatted_msg}{Colors.RESET}"
            print(colored_msg)

        # File output
        if self.file_output:
            self.file_output.write(formatted_msg + "\n")
            self.file_output.flush()

    # ------------------------------------------------------------------
    # Public helper methods
    # ------------------------------------------------------------------
    def debug(self, message: str) -> None:
        self._log(LogLevel.DEBUG, message)

    def info(self, message: str) -> None:
        self._log(LogLevel.INFO, message)

    def warning(self, message: str) -> None:
        self._log(LogLevel.WARNING, message)

    def error(self, message: str) -> None:
        self._log(LogLevel.ERROR, message)

    def critical(self, message: str) -> None:
        self._log(LogLevel.CRITICAL, message)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self) -> None:
        if self.file_output:
            try:
                self.file_output.close()
            except Exception:
                pass


# Singleton instance used throughout the project
logger = SimpleLogger()


if __name__ == "__main__":  # pragma: no cover - manual test only
    test_logger = SimpleLogger(name="test_logger", log_level=LogLevel.DEBUG)
    test_logger.debug("This is a DEBUG message")
    test_logger.info("This is an INFO message")
    test_logger.warning("This is a WARNING message")
    test_logger.error("This is an ERROR message")
    test_logger.critical("This is a CRITICAL message")
