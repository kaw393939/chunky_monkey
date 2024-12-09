"""Logging configuration for document processor."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Configure logging for the application."""
    if format_string is None:
        format_string = (
            "%(asctime)s [%(levelname)s] %(name)s: "
            "%(message)s (%(filename)s:%(lineno)d)"
        )

    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create logger
    logger = logging.getLogger("document_processor")
    logger.setLevel(level)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        log_dir = log_file.parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(f"document_processor.{name}")

# Create default logger
default_log_file = Path("logs") / f"document_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger = setup_logging(log_file=default_log_file)