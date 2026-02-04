"""
2api.ai - Structured JSON Logging

Production-grade structured logging with automatic context injection.

Features:
- JSON-formatted logs for easy parsing
- Automatic trace context injection (trace_id, span_id, request_id)
- Log levels configurable via environment
- Correlation across services
- Sensitive data redaction

Usage:
    from src.observability.logging import setup_logging, get_logger

    # Setup at startup
    setup_logging(level="INFO")

    # Get logger
    logger = get_logger(__name__)
    logger.info("Processing request", extra={"user_id": "123"})

Output:
    {"timestamp": "2024-01-15T10:30:00Z", "level": "INFO", "logger": "__main__",
     "message": "Processing request", "user_id": "123", "trace_id": "abc123",
     "span_id": "def456", "request_id": "req_xyz"}
"""

import os
import sys
import json
import logging
import time
from typing import Optional, Dict, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field
from contextvars import ContextVar
from functools import wraps

# Context variables for correlation IDs
_request_context: ContextVar[Optional["LogContext"]] = ContextVar("log_context", default=None)


@dataclass
class LogContext:
    """
    Logging context with correlation IDs.

    Thread-safe using contextvars.
    """
    request_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    tenant_id: str = ""
    api_key_prefix: str = ""
    provider: str = ""
    model: str = ""
    endpoint: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def get_current(cls) -> Optional["LogContext"]:
        """Get current log context."""
        return _request_context.get()

    @classmethod
    def set_current(cls, ctx: "LogContext"):
        """Set current log context."""
        _request_context.set(ctx)

    @classmethod
    def clear(cls):
        """Clear current log context."""
        _request_context.set(None)

    def update(self, **kwargs):
        """Update context fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        result = {}
        if self.request_id:
            result["request_id"] = self.request_id
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.span_id:
            result["span_id"] = self.span_id
        if self.tenant_id:
            result["tenant_id"] = self.tenant_id
        if self.api_key_prefix:
            result["api_key_prefix"] = self.api_key_prefix
        if self.provider:
            result["provider"] = self.provider
        if self.model:
            result["model"] = self.model
        if self.endpoint:
            result["endpoint"] = self.endpoint
        result.update(self.extra)
        return result


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter with automatic context injection.

    Output format:
    {
        "timestamp": "2024-01-15T10:30:00.123456Z",
        "level": "INFO",
        "logger": "module.name",
        "message": "Log message",
        "request_id": "req_abc123",
        "trace_id": "trace_def456",
        "span_id": "span_789",
        ... additional fields
    }
    """

    # Fields to redact from logs
    SENSITIVE_FIELDS = {
        "password", "secret", "token", "api_key", "apikey",
        "authorization", "auth", "credential", "private_key",
    }

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_logger: bool = True,
        include_location: bool = False,  # filename:lineno
        redact_sensitive: bool = True,
    ):
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_logger = include_logger
        self.include_location = include_location
        self.redact_sensitive = redact_sensitive

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {}

        # Core fields
        if self.include_timestamp:
            log_data["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self.include_level:
            log_data["level"] = record.levelname

        if self.include_logger:
            log_data["logger"] = record.name

        # Message
        log_data["message"] = record.getMessage()

        # Location (optional)
        if self.include_location:
            log_data["location"] = f"{record.filename}:{record.lineno}"

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Inject context from contextvars
        ctx = LogContext.get_current()
        if ctx:
            log_data.update(ctx.to_dict())

        # Add extra fields from record
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in {
                    "name", "msg", "args", "created", "filename",
                    "funcName", "levelname", "levelno", "lineno",
                    "module", "msecs", "pathname", "process",
                    "processName", "relativeCreated", "stack_info",
                    "exc_info", "exc_text", "thread", "threadName",
                    "message", "taskName",
                }:
                    # Redact sensitive fields
                    if self.redact_sensitive and self._is_sensitive(key):
                        value = "[REDACTED]"
                    log_data[key] = value

        return json.dumps(log_data, default=str, ensure_ascii=False)

    def _is_sensitive(self, field_name: str) -> bool:
        """Check if field name indicates sensitive data."""
        field_lower = field_name.lower()
        return any(sensitive in field_lower for sensitive in self.SENSITIVE_FIELDS)


class StructuredLogger:
    """
    Structured logger wrapper with convenience methods.

    Automatically injects context and supports structured extra fields.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    @property
    def name(self) -> str:
        return self._logger.name

    def _log(self, level: int, msg: str, *args, **kwargs):
        """Internal log method with context injection."""
        extra = kwargs.pop("extra", {})

        # Get context
        ctx = LogContext.get_current()
        if ctx:
            extra.update(ctx.to_dict())

        # Merge any additional kwargs as extra
        for key, value in list(kwargs.items()):
            if key not in {"exc_info", "stack_info", "stacklevel"}:
                extra[key] = kwargs.pop(key)

        kwargs["extra"] = extra
        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        """Log exception with traceback."""
        kwargs["exc_info"] = True
        self._log(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, msg, *args, **kwargs)

    def with_context(self, **context_fields) -> "StructuredLogger":
        """Return logger with additional context fields bound."""
        # This creates a temporary context update
        ctx = LogContext.get_current()
        if ctx:
            ctx.update(**context_fields)
        return self


# Module-level state
_logging_configured = False
_log_level = logging.INFO


def setup_logging(
    level: Union[str, int] = "INFO",
    json_output: bool = True,
    include_location: bool = False,
    redact_sensitive: bool = True,
) -> None:
    """
    Setup structured logging.

    Call once at application startup.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Use JSON formatter (True) or standard formatter (False)
        include_location: Include filename:lineno in logs
        redact_sensitive: Redact sensitive fields like passwords
    """
    global _logging_configured, _log_level

    # Parse level
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    _log_level = level

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Set formatter
    if json_output:
        formatter = JSONFormatter(
            include_location=include_location,
            redact_sensitive=redact_sensitive,
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger.

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    global _logging_configured

    # Auto-configure if needed
    if not _logging_configured:
        level = os.getenv("LOG_LEVEL", "INFO")
        json_output = os.getenv("LOG_FORMAT", "json").lower() == "json"
        setup_logging(level=level, json_output=json_output)

    logger = logging.getLogger(name)
    return StructuredLogger(logger)


def log_context(**fields):
    """
    Decorator to add context to all logs within a function.

    Usage:
        @log_context(operation="process_request")
        async def handle_request(...):
            logger.info("Processing")  # Will include operation="process_request"
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            ctx = LogContext.get_current() or LogContext()
            ctx.update(**fields)
            LogContext.set_current(ctx)
            try:
                return await func(*args, **kwargs)
            finally:
                # Don't clear - let parent context remain
                pass

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            ctx = LogContext.get_current() or LogContext()
            ctx.update(**fields)
            LogContext.set_current(ctx)
            try:
                return func(*args, **kwargs)
            finally:
                pass

        if asyncio_available() and is_coroutine_function(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def asyncio_available() -> bool:
    """Check if asyncio is available."""
    try:
        import asyncio
        return True
    except ImportError:
        return False


def is_coroutine_function(func) -> bool:
    """Check if function is a coroutine."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


# Performance logging helper
class TimedOperation:
    """
    Context manager for timing operations.

    Usage:
        with TimedOperation("database_query", logger) as timer:
            result = await db.query(...)
        # Logs: "database_query completed" with duration_ms
    """

    def __init__(
        self,
        operation: str,
        logger: Optional[StructuredLogger] = None,
        log_level: int = logging.DEBUG,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.operation = operation
        self.logger = logger or get_logger("timed_operation")
        self.log_level = log_level
        self.extra = extra or {}
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None

    def __enter__(self) -> "TimedOperation":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000

        log_extra = {
            "operation": self.operation,
            "duration_ms": round(self.duration_ms, 2),
            **self.extra,
        }

        if exc_type:
            log_extra["error"] = str(exc_val)
            self.logger._log(
                logging.ERROR,
                f"{self.operation} failed",
                extra=log_extra,
            )
        else:
            self.logger._log(
                self.log_level,
                f"{self.operation} completed",
                extra=log_extra,
            )

    async def __aenter__(self) -> "TimedOperation":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)
