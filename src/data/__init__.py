"""
Data handling module for VLM training data.
"""

# This file makes the data directory a Python package
# Import dataset classes here when they're created
from .dataset import VLMDataset  # noqa: F401

__all__ = ['VLMDataset']
