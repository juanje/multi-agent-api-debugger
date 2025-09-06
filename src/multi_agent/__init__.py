"""
Multi-agent system with LangGraph.

This package implements an intelligent multi-agent chatbot that can:
- Execute simple operations (calculator, time, RAG)
- Execute parallel operations
- Execute sequential operations with intermediate decisions
"""

from .cli import main
from .chat import run_chat
from .graph import app
from .state import ChatState

__version__ = "0.1.0"
__all__ = ["main", "run_chat", "app", "ChatState"]
