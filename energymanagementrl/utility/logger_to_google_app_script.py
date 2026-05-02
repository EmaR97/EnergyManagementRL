import logging
import sys
from typing import Optional

import requests


class RemotePostHandler(logging.Handler):
    """Logging handler that sends log records to a remote HTTP endpoint."""

    def __init__(self, webhook_url: str, sheet_key: Optional[str] = None):
        super().__init__()
        self.webhook_url = webhook_url
        self.sheet_key = sheet_key

    def emit(self, record: logging.LogRecord) -> None:
        """Send log record to the remote endpoint."""
        try:
            log_entry = {'name': record.name, 'level': record.levelname, 'message': record.message,
                         'sheetKey': self.sheet_key or 'default', }

            response = requests.post(self.webhook_url, json=log_entry, timeout=5)

            response.raise_for_status()

        except Exception as e:
            # Fall back to stderr if remote logging fails
            print(f"Failed to send log to remote endpoint: {e}", file=sys.stderr)


class LoggerToRemotePost:
    """Context manager for logging to remote POST endpoint."""

    def __init__(self, logger: logging.Logger, webhook_url: str, sheet_key: Optional[str] = None):
        """
        Initialize the remote POST logger.

        Args:
            logger: The logger instance to add the handler to
            webhook_url: The URL of the Google Apps Script webhook
            sheet_key: Optional key to determine which sheet to log to
        """
        self.logger = logger
        self.webhook_url = webhook_url
        self.sheet_key = sheet_key
        self.handler = None
        self.handler_added = False

    def __enter__(self):
        """Add the remote POST handler to the logger."""
        # Check if handler already exists
        if not any(
                isinstance(h, RemotePostHandler) and h.webhook_url == self.webhook_url for h in self.logger.handlers):
            self.handler = RemotePostHandler(self.webhook_url, self.sheet_key)
            self.handler.setLevel(logging.INFO)
            self.logger.addHandler(self.handler)
            self.handler_added = True

        return self.logger

    def __exit__(self, exc_type, exc_value, traceback):
        """Remove the handler from the logger."""
        if self.handler_added and self.handler:
            self.logger.removeHandler(self.handler)
