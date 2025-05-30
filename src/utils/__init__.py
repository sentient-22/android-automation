"""Utility functions for the Android automation project."""

# Import key functions for easier access
from .logger import setup_logger, get_timestamp, logger  # noqa: F401
from .helpers import capture_screenshot, get_ui_hierarchy  # noqa: F401

__all__ = ['setup_logger', 'get_timestamp', 'logger', 'capture_screenshot', 'get_ui_hierarchy']
