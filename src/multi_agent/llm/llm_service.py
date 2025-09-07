"""
LLM Service for managing Ollama connections and model interactions.

This module provides a centralized service for managing LLM connections,
handling model initialization, and providing a unified interface for
all agent interactions with language models.
"""

from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from .llm_config import AgentType, get_agent_config, LLMConfig

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """Exception raised for LLM service errors."""

    pass


class LLMService:
    """Centralized service for managing LLM connections and interactions."""

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        """Initialize the LLM service.

        Args:
            ollama_host: The host URL for the Ollama server
        """
        self.ollama_host = ollama_host
        self._models: Dict[AgentType, ChatOllama] = {}
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all configured models."""
        if self._initialized:
            return

        logger.info("Initializing LLM service...")

        # Initialize models for each agent type
        for agent_type in AgentType:
            try:
                config = get_agent_config(agent_type)
                model = self._create_model(config.llm_config)
                self._models[agent_type] = model
                logger.info(
                    f"Initialized model for {agent_type.value}: {config.llm_config.model_name}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize model for {agent_type.value}: {e}")
                raise LLMServiceError(
                    f"Failed to initialize model for {agent_type.value}: {e}"
                )

        self._initialized = True
        logger.info("LLM service initialized successfully")

    def _create_model(self, config: LLMConfig) -> ChatOllama:
        """Create a ChatOllama model instance.

        Args:
            config: The LLM configuration

        Returns:
            Configured ChatOllama instance
        """
        try:
            model = ChatOllama(
                model=config.model_name,
                base_url=self.ollama_host,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                repeat_penalty=config.repeat_penalty,
                num_ctx=config.num_ctx,
            )
            return model
        except Exception as e:
            logger.error(f"Failed to create model {config.model_name}: {e}")
            raise LLMServiceError(f"Failed to create model {config.model_name}: {e}")

    def get_model(self, agent_type: AgentType) -> ChatOllama:
        """Get the model for a specific agent type.

        Args:
            agent_type: The type of agent

        Returns:
            The ChatOllama model for the agent

        Raises:
            LLMServiceError: If the service is not initialized or model not found
        """
        # Lazy initialization: create model if it doesn't exist
        if agent_type not in self._models:
            try:
                config = get_agent_config(agent_type)
                model = self._create_model(config.llm_config)
                self._models[agent_type] = model
                logger.info(
                    f"Lazy initialized model for {agent_type.value}: {config.llm_config.model_name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to lazy initialize model for {agent_type.value}: {e}"
                )
                raise LLMServiceError(
                    f"Failed to lazy initialize model for {agent_type.value}: {e}"
                )

        return self._models[agent_type]

    def generate_response(
        self, agent_type: AgentType, messages: List[BaseMessage], **kwargs
    ) -> str:
        """Generate a response using the specified agent's model.

        Args:
            agent_type: The type of agent to use
            messages: List of messages for the conversation
            **kwargs: Additional parameters for the model

        Returns:
            The generated response text
        """
        model = self.get_model(agent_type)

        try:
            response = model.invoke(messages, **kwargs)
            content = response.content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Handle list of content blocks
                return str(content)
            else:
                return str(content)
        except Exception as e:
            logger.error(f"Failed to generate response for {agent_type.value}: {e}")
            raise LLMServiceError(
                f"Failed to generate response for {agent_type.value}: {e}"
            )

    def generate_with_system_prompt(
        self,
        agent_type: AgentType,
        user_message: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate a response with a system prompt.

        Args:
            agent_type: The type of agent to use
            user_message: The user's message
            system_prompt: Optional system prompt (uses agent's default if None)
            **kwargs: Additional parameters for the model

        Returns:
            The generated response text
        """
        if system_prompt is None:
            config = get_agent_config(agent_type)
            system_prompt = config.system_prompt

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]

        return self.generate_response(agent_type, messages, **kwargs)

    def generate_with_template(
        self,
        agent_type: AgentType,
        template: str,
        input_variables: Dict[str, Any],
        **kwargs,
    ) -> str:
        """Generate a response using a template.

        Args:
            agent_type: The type of agent to use
            template: The prompt template
            input_variables: Variables to substitute in the template
            **kwargs: Additional parameters for the model

        Returns:
            The generated response text
        """
        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(**input_variables)
        return self.generate_response(agent_type, messages, **kwargs)

    def stream_response(
        self, agent_type: AgentType, messages: List[BaseMessage], **kwargs
    ):
        """Stream a response from the specified agent's model.

        Args:
            agent_type: The type of agent to use
            messages: List of messages for the conversation
            **kwargs: Additional parameters for the model

        Yields:
            Chunks of the response
        """
        model = self.get_model(agent_type)

        try:
            for chunk in model.stream(messages, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"Failed to stream response for {agent_type.value}: {e}")
            raise LLMServiceError(
                f"Failed to stream response for {agent_type.value}: {e}"
            )

    def get_model_info(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get information about a specific model.

        Args:
            agent_type: The type of agent

        Returns:
            Dictionary with model information
        """
        config = get_agent_config(agent_type)

        return {
            "agent_type": agent_type.value,
            "model_name": config.llm_config.model_name,
            "temperature": config.llm_config.temperature,
            "top_p": config.llm_config.top_p,
            "top_k": config.llm_config.top_k,
            "repeat_penalty": config.llm_config.repeat_penalty,
            "num_ctx": config.llm_config.num_ctx,
            "system_prompt": config.system_prompt,
        }

    def get_all_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models.

        Returns:
            Dictionary with information about all models
        """
        return {
            agent_type.value: self.get_model_info(agent_type)
            for agent_type in AgentType
        }


# Global LLM service instance
_llm_service: Optional[LLMService] = None


def get_llm_service(ollama_host: str = "http://localhost:11434") -> LLMService:
    """Get the global LLM service instance.

    Args:
        ollama_host: The host URL for the Ollama server

    Returns:
        The global LLM service instance
    """
    global _llm_service

    if _llm_service is None:
        _llm_service = LLMService(ollama_host)
        # Don't initialize all models upfront - use lazy loading
        logger.info("LLM service created (models will be initialized on demand)")

    return _llm_service


def generate_agent_response(
    agent_type: AgentType,
    user_message: str,
    system_prompt: Optional[str] = None,
    ollama_host: str = "http://localhost:11434",
    **kwargs,
) -> str:
    """Generate a response using the specified agent.

    Args:
        agent_type: The type of agent to use
        user_message: The user's message
        system_prompt: Optional system prompt (uses agent's default if None)
        ollama_host: The host URL for the Ollama server
        **kwargs: Additional parameters for the model

    Returns:
        The generated response text
    """
    service = get_llm_service(ollama_host)
    return service.generate_with_system_prompt(
        agent_type, user_message, system_prompt, **kwargs
    )
