"""
Memory module for Long Term Memory (LTM) functionality.

This module provides persistent storage and retrieval of system interactions,
including Q&A sessions, debugging analyses, and API operations.
"""

from .schema import MemoryEntry, MemoryType
from .ltm_service import LTMService, get_ltm_service
from .vector_store import VectorStoreService
from .config import (
    LTM_CONFIDENCE_THRESHOLD,
    LTM_SEARCH_LIMIT_DEFAULT,
    LTM_RECENT_LIMIT_DEFAULT,
    LTM_DEFAULT_DIRECTORY,
)

__all__ = [
    "MemoryEntry",
    "MemoryType",
    "LTMService",
    "get_ltm_service",
    "VectorStoreService",
    "LTM_CONFIDENCE_THRESHOLD",
    "LTM_SEARCH_LIMIT_DEFAULT",
    "LTM_RECENT_LIMIT_DEFAULT",
    "LTM_DEFAULT_DIRECTORY",
]
