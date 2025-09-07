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
from .graph import get_graph, GraphState
from .llm import AgentType, get_agent_config, LLMService, get_llm_service
from .llm.llm_config import LLMConfig, AgentLLMConfig
from .llm.llm_service import generate_agent_response
from .llm.prompts import get_agent_prompt, get_all_prompts, AgentPrompts

__version__ = "0.1.0"
__all__ = [
    "main",
    "run_chat",
    "get_graph",
    "GraphState",
    "AgentType",
    "LLMConfig",
    "AgentLLMConfig",
    "get_agent_config",
    "LLMService",
    "get_llm_service",
    "generate_agent_response",
    "get_agent_prompt",
    "get_all_prompts",
    "AgentPrompts",
]
