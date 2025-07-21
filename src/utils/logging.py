# src/utils/logging.py
import structlog
import logging
import sys
from typing import Dict, Any

def configure_logging(log_level: str = "INFO") -> None:
    """Configure structured logging"""
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

class CorrelationIDFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Add correlation ID if available
        if not hasattr(record, 'correlation_id'):
            record.correlation_id = 'none'
        return True

def get_logger(name: str) -> structlog.BoundLogger:
    """Get configured logger"""
    return structlog.get_logger(name)