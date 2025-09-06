"""
Multi-agent system with LangGraph following the architecture design.

This package implements an intelligent multi-agent system that can:
- Execute API operations (run jobs, get results, check status)
- Debug errors with root cause analysis
- Answer questions using knowledge base
- Synthesize responses for users
"""

from .cli import main
from .chat import run_chat
from .graph import app
from .state import GraphState

__version__ = "0.1.0"
__all__ = ["main", "run_chat", "app", "GraphState"]
