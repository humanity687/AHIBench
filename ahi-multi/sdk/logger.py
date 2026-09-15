import logging
from logging.handlers import RotatingFileHandler
import os
import time
from typing import Any, Dict, List, Optional


class Logger:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_file: str = "ahi_system.log", audit_file: str = "ahi_audit.log",
                 debug: bool = True):
        if not self._initialized:
            self.log_file = log_file
            self.audit_file = audit_file
            self.debug_mode = debug
            self._setup_logger()
            self._initialized = True

    def _setup_logger(self):
        self.logger = logging.getLogger("AHI")
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        self.logger.propagate = False

        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=100*1024*1024, backupCount=10, encoding="utf-8"
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        audit_handler = RotatingFileHandler(
            self.audit_file, maxBytes=100*1024*1024, backupCount=10, encoding="utf-8"
        )
        audit_handler.setLevel(logging.WARNING)
        audit_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(audit_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message: str, **kwargs):
        self.logger.debug(message)

    def info(self, message: str, **kwargs):
        self.logger.info(message)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message)

    def error(self, message: str, **kwargs):
        self.logger.error(message)

    def critical(self, message: str, **kwargs):
        self.logger.critical(message)
