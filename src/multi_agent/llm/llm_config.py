"""
LLM Configuration module for the multi-agent system.

This module defines the configuration for different LLM models used by each agent
in the system, optimized for their specific roles and responsibilities.
"""

from __future__ import annotations
from typing import Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class AgentType(str, Enum):
    """Enumeration of agent types in the system."""

    SUPERVISOR = "supervisor"
    API_OPERATOR = "api_operator"
    DEBUGGER = "debugger"
    KNOWLEDGE_ASSISTANT = "knowledge_assistant"
    RESPONSE_SYNTHESIZER = "response_synthesizer"


class LLMConfig(BaseModel):
    """Configuration for a specific LLM model."""

    model_name: str = Field(..., description="Name of the Ollama model")
    temperature: float = Field(
        default=0.0, ge=0.0, le=2.0, description="Temperature for generation"
    )
    max_tokens: Optional[int] = Field(
        default=None, ge=1, description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    top_k: int = Field(default=40, ge=1, description="Top-k sampling parameter")
    repeat_penalty: float = Field(default=1.1, ge=0.0, description="Repeat penalty")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    validate_model: bool = Field(
        default=True, description="Whether to validate model on initialization"
    )
    num_ctx: int = Field(
        default=2048, ge=512, le=8192, description="Context window size"
    )
    num_batch: int = Field(
        default=512, ge=128, le=2048, description="Batch size for processing"
    )


class AgentLLMConfig(BaseModel):
    """Configuration for an agent's LLM setup."""

    agent_type: AgentType = Field(..., description="Type of agent")
    llm_config: LLMConfig = Field(..., description="LLM configuration")
    system_prompt: str = Field(..., description="System prompt for the agent")
    description: str = Field(..., description="Description of the agent's role")


# Default LLM configurations for each agent type
AGENT_LLM_CONFIGS: Dict[AgentType, AgentLLMConfig] = {
    AgentType.SUPERVISOR: AgentLLMConfig(
        agent_type=AgentType.SUPERVISOR,
        llm_config=LLMConfig(
            model_name="llama3.1:8b",
            temperature=0.0,  # Deterministic routing decisions
            max_tokens=512,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            timeout=30,
            num_ctx=2048,  # Medium context for routing decisions
            num_batch=512,
        ),
        system_prompt="""You are a Supervisor agent responsible for orchestrating multi-agent workflows. 
Your primary responsibilities are:
1. Analyze user requests to understand intent and context
2. Route requests to the appropriate specialist agent
3. Manage task execution flow and dependencies
4. Make routing decisions based on conversation history and current state

Available agents:
- api_operator: For API operations, job management, and system interactions
- debugger: For error analysis, root cause investigation, and troubleshooting
- knowledge_assistant: For answering questions and providing information
- response_synthesizer: For formatting and presenting final responses

Always provide clear reasoning for your routing decisions and maintain context awareness.""",
        description="Central orchestrator for routing and workflow management",
    ),
    AgentType.API_OPERATOR: AgentLLMConfig(
        agent_type=AgentType.API_OPERATOR,
        llm_config=LLMConfig(
            model_name="llama3.2:3b",
            temperature=0.0,  # Deterministic tool execution
            max_tokens=256,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            timeout=15,
            num_ctx=1024,  # Small context for simple tool execution
            num_batch=256,
        ),
        system_prompt="""You are an API Operator agent responsible for executing API operations.
Your primary responsibilities are:
1. Execute API calls based on instructions from the Supervisor
2. Handle authentication and authorization for different tool categories
3. Report raw results (success data or error messages) back to the state
4. Execute predefined workflows and tool sequences

Available tools:
- list_public_jobs: List available jobs
- run_job: Execute a specific job
- get_job_results: Retrieve job results
- check_system_status: Check system health

Execute tools accurately and report results without interpretation.""",
        description="Tool-using agent for API operations and system interactions",
    ),
    AgentType.DEBUGGER: AgentLLMConfig(
        agent_type=AgentType.DEBUGGER,
        llm_config=LLMConfig(
            model_name="qwen2.5-coder:7b",
            temperature=0.3,  # Some creativity for analysis
            max_tokens=1024,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            timeout=45,
            num_ctx=4096,  # Large context for analyzing logs and complex data
            num_batch=1024,
        ),
        system_prompt="""You are a Debugger agent specialized in error analysis and troubleshooting.
Your primary responsibilities are:
1. Analyze error logs and failure data
2. Formulate hypotheses about root causes
3. Use available tools to gather additional information
4. Provide structured root cause analysis
5. Suggest solutions and remediation steps

Analysis approach:
- Think step-by-step through the problem
- Consider system architecture and dependencies
- Look for common error patterns
- Correlate multiple data sources
- Provide actionable insights

Always structure your analysis clearly and provide evidence for your conclusions.""",
        description="Specialized agent for error analysis and root cause investigation",
    ),
    AgentType.KNOWLEDGE_ASSISTANT: AgentLLMConfig(
        agent_type=AgentType.KNOWLEDGE_ASSISTANT,
        llm_config=LLMConfig(
            model_name="llama3-chatqa:8b",
            temperature=0.7,  # More conversational tone
            max_tokens=768,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            timeout=30,
            num_ctx=3072,  # Large context for Q&A with retrieved documents
            num_batch=768,
        ),
        system_prompt="""You are a Knowledge Assistant agent responsible for answering questions and providing information.
Your primary responsibilities are:
1. Answer user questions using available knowledge bases
2. Use RAG (Retrieval-Augmented Generation) to find relevant information
3. Access long-term memory for context from past conversations
4. Provide helpful, accurate, and well-structured responses

Guidelines:
- Base answers strictly on provided sources and context
- If information is not available, clearly state this
- Use a helpful and professional tone
- Structure responses clearly with examples when appropriate
- Reference sources when available

Always prioritize accuracy over completeness.""",
        description="Conversational agent for knowledge queries and information retrieval",
    ),
    AgentType.RESPONSE_SYNTHESIZER: AgentLLMConfig(
        agent_type=AgentType.RESPONSE_SYNTHESIZER,
        llm_config=LLMConfig(
            model_name="gemma3:4b",
            temperature=0.6,  # Balanced creativity and consistency
            max_tokens=768,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            timeout=30,
            num_ctx=3072,  # Large context for synthesizing multiple data sources
            num_batch=768,
        ),
        system_prompt="""You are a Response Synthesizer agent responsible for formatting and presenting final responses.
Your primary responsibilities are:
1. Synthesize raw data from the Graph State into natural language
2. Format responses appropriately using Markdown when helpful
3. Create clear, concise, and user-friendly messages
4. Consolidate key information for long-term memory storage
5. Ensure responses are professional and actionable

Formatting guidelines:
- Use clear headings and structure
- Include relevant details without overwhelming the user
- Highlight key findings and next steps
- Use appropriate formatting (lists, code blocks, etc.)
- Maintain a professional yet approachable tone

Focus on clarity and usability in all responses.""",
        description="Agent for formatting and presenting final user responses",
    ),
}


def get_agent_config(agent_type: AgentType) -> AgentLLMConfig:
    """Get the LLM configuration for a specific agent type.

    Args:
        agent_type: The type of agent to get configuration for

    Returns:
        The LLM configuration for the specified agent

    Raises:
        ValueError: If the agent type is not supported
    """
    if agent_type not in AGENT_LLM_CONFIGS:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    return AGENT_LLM_CONFIGS[agent_type]


def get_all_agent_configs() -> Dict[AgentType, AgentLLMConfig]:
    """Get all available agent LLM configurations.

    Returns:
        Dictionary mapping agent types to their configurations
    """
    return AGENT_LLM_CONFIGS.copy()


def update_agent_config(agent_type: AgentType, config: AgentLLMConfig) -> None:
    """Update the configuration for a specific agent type.

    Args:
        agent_type: The type of agent to update
        config: The new configuration for the agent
    """
    AGENT_LLM_CONFIGS[agent_type] = config


def validate_model_availability() -> Dict[str, bool]:
    """Validate that all configured models are available in Ollama.

    Returns:
        Dictionary mapping model names to their availability status
    """
    # This will be implemented when we create the LLMService
    # For now, return a placeholder
    models = set()
    for config in AGENT_LLM_CONFIGS.values():
        models.add(config.llm_config.model_name)

    return {model: True for model in models}  # Placeholder
