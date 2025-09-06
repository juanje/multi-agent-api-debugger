"""
  Shared state of the multi-agent system.
"""

from __future__ import annotations
from typing import Literal, Optional, List, Dict, Any
from langgraph.graph import MessagesState


class ChatState(MessagesState):
    """Chat state with additional fields for multi-agent flow."""

    route: Optional[Literal["tools", "rag", "parallel", "sequential", "done"]]
    operations: Optional[List[str]]
    steps: Optional[List[Dict[str, Any]]]
    results: Optional[List[str]]
    current_step: Optional[int]
    pending_tasks: Optional[List[str]]
