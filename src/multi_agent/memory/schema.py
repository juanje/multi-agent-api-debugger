"""
Memory schema definitions for Long Term Memory (LTM) system.

This module defines the data structures used to store different types
of memories in the vector database.
"""

from __future__ import annotations
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Types of memories that can be stored in LTM."""

    QA_SESSION = "qa_session"
    DEBUG_ANALYSIS = "debug_analysis"
    API_OPERATION = "api_operation"
    KNOWLEDGE_QUERY = "knowledge_query"


class MemoryEntry(BaseModel):
    """
    A memory entry stored in the Long Term Memory system.

    This represents a single interaction or piece of knowledge that
    can be stored and retrieved for future reference.
    """

    id: Optional[str] = None
    type: MemoryType
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    user_query: str
    system_response: str
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Specific fields for different memory types
    error_code: Optional[str] = None  # For debug analysis
    confidence_level: Optional[str] = None  # For debug analysis
    severity: Optional[str] = None  # For debug analysis
    api_operation: Optional[str] = None  # For API operations
    success: Optional[bool] = None  # For API operations
    related_topics: List[str] = Field(default_factory=list)  # For knowledge queries

    def get_searchable_text(self) -> str:
        """Get the text that should be used for semantic search."""
        parts = [self.user_query, self.system_response]

        # Add type-specific context
        if self.error_code:
            parts.append(f"Error: {self.error_code}")
        if self.api_operation:
            parts.append(f"API Operation: {self.api_operation}")
        if self.related_topics:
            parts.append(f"Topics: {', '.join(self.related_topics)}")

        return " ".join(parts)

    def get_metadata_dict(self) -> Dict[str, Any]:
        """Get metadata dictionary for vector store."""
        metadata = {
            "type": self.type.value,
            "timestamp": self.timestamp,
            **self.metadata,
        }

        # Add non-None specific fields
        if self.error_code:
            metadata["error_code"] = self.error_code
        if self.confidence_level:
            metadata["confidence_level"] = self.confidence_level
        if self.severity:
            metadata["severity"] = self.severity
        if self.api_operation:
            metadata["api_operation"] = self.api_operation
        if self.success is not None:
            metadata["success"] = self.success
        if self.related_topics:
            metadata["related_topics"] = self.related_topics

        return metadata


class SearchResult(BaseModel):
    """Result from a memory search operation."""

    memory: MemoryEntry
    similarity_score: float
    rank: int
