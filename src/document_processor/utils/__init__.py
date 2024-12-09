"""Utility functions and classes for document processor."""

from .config import Config, MODEL_CONFIGS, DEFAULT_MODEL_NAME
from .logging import get_logger, setup_logging

__all__ = [
    'Config',
    'MODEL_CONFIGS',
    'DEFAULT_MODEL_NAME',
    'get_logger',
    'setup_logging',
]