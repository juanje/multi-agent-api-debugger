"""
Routing intelligence for the Supervisor agent using LLMs.

This module contains routing logic that uses LLMs to make intelligent decisions
about which agent should handle the next step in the workflow.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from .state import GraphState
from ..llm import AgentType, get_llm_service, should_use_mocks
from ..llm.llm_service import LLMServiceError
from ..llm.llm_mocks import (
    get_mock_route_determination,
    get_mock_todo_list_creation,
    get_mock_api_operations_extraction,
)
from ..llm.prompts import get_agent_prompt

logger = logging.getLogger(__name__)


async def determine_route(state: GraphState) -> str:
    """Determine the next route based on current state using LLM.

    Args:
        state: Current graph state

    Returns:
        Next route to take (agent name or "done")
    """
    if should_use_mocks():
        return get_mock_route_determination(state)

    try:
        service = get_llm_service()

        # Build context for the LLM
        context: Dict[str, Any] = {
            "current_state": {
                "has_final_response": bool(state.get("final_response")),
                "has_results": bool(state.get("results")),
                "has_root_cause_analysis": bool(state.get("root_cause_analysis")),
                "has_todo_list": bool(state.get("todo_list")),
                "has_error_info": bool(state.get("error_info")),
                "message_count": len(state.get("messages", [])),
            }
        }

        # Add last message content if available
        if state.get("messages"):
            last_message = state["messages"][-1]
            content = (
                last_message.content
                if hasattr(last_message, "content")
                else str(last_message)
            )
            if isinstance(content, list):
                text = ""
                for item in content:
                    if isinstance(item, str):
                        text = item
                        break
            else:
                text = content or ""
            context["last_message"] = text

        prompt = f"""You are a Supervisor agent responsible for routing requests to the appropriate specialized agent.

Current state context:
{json.dumps(context, indent=2)}

Available agents:
- api_operator: Handles API operations (run jobs, get results, list jobs, check system status)
- debugger: Analyzes errors, logs, and provides root cause analysis
- knowledge_assistant: Answers questions and provides information
- response_synthesizer: Formats and presents final responses to users

CRITICAL ROUTING RULES:
1. If there's a final_response, return "done"
2. If there are results or root_cause_analysis to synthesize, route to "response_synthesizer"
3. If there are pending todo items, route to "api_operator"
4. If there's error_info but no root_cause_analysis, route to "debugger"
5. Otherwise, analyze the last message content and route based on intent:
   - Questions (what, how, why, when, where, ?) → "knowledge_assistant"
   - Debug/error requests (debug, depura, error, fail, investigate) → "debugger"
   - API operations (run, get, list, check) → "api_operator"
   - Default → "api_operator"

SPECIAL CASE: If the user says "depura" or "debug" with a job ID (like "depura job_003"), ALWAYS route to "debugger".

IMPORTANT: You must respond with ONLY the agent name or "done". Do not include any JSON, explanations, or other text. Just the agent name."""

        response_content = service.generate_with_system_prompt(
            AgentType.SUPERVISOR, prompt, get_agent_prompt("supervisor")
        )

        # Log the raw LLM response for debugging
        logger.info(f"LLM routing response: '{response_content}'")

        # Clean and validate response
        response_clean = response_content.strip()

        # Try to parse as JSON first
        try:
            json_response = json.loads(response_clean)
            if isinstance(json_response, dict) and "next_agent" in json_response:
                route = json_response["next_agent"].lower()
            else:
                route = response_clean.lower()
        except json.JSONDecodeError:
            # Not JSON, treat as plain text
            route = response_clean.lower()

        valid_routes = [
            "api_operator",
            "debugger",
            "knowledge_assistant",
            "response_synthesizer",
            "done",
        ]

        if route in valid_routes:
            return route
        else:
            logger.warning(
                f"Invalid route from LLM: {response_clean}, defaulting to api_operator"
            )
            return "api_operator"

    except LLMServiceError as e:
        logger.error(f"LLMServiceError in determine_route: {e}")
        return get_mock_route_determination(state)
    except Exception as e:
        logger.error(f"Unexpected error in determine_route: {e}")
        return get_mock_route_determination(state)


async def create_todo_list(state: GraphState) -> List[str]:
    """Create todo list based on user request using LLM.

    Args:
        state: Current graph state

    Returns:
        List of todo items (API operations)
    """
    if should_use_mocks():
        return get_mock_todo_list_creation(state)

    try:
        service = get_llm_service()

        # Extract message content
        if not state.get("messages"):
            return []

        last_message = state["messages"][-1]
        content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )

        if isinstance(content, list):
            text = ""
            for item in content:
                if isinstance(item, str):
                    text = item
                    break
        else:
            text = content or ""

        prompt = f"""You are a Supervisor agent that creates todo lists for API operations.

User request: "{text}"

Available API operations:
- list_public_jobs: List all available jobs
- run_job:job_name=<name>: Run a specific job (e.g., data_processing, image_analysis, report_generation)
- get_job_results:job_id=<id>: Get results for a specific job
- check_system_status: Check overall system status

Extract the API operations needed to fulfill this request. Return a JSON array of operation strings.

Examples:
- "List all jobs" → ["list_public_jobs"]
- "Run data processing job" → ["run_job:job_name=data_processing"]
- "Get results for job_123" → ["get_job_results:job_id=job_123"]
- "Check system status" → ["check_system_status"]

Respond with ONLY a JSON array - no explanation needed."""

        response_content = service.generate_with_system_prompt(
            AgentType.SUPERVISOR, prompt, get_agent_prompt("supervisor")
        )

        # Parse JSON response
        try:
            operations = json.loads(response_content.strip())
            if isinstance(operations, list):
                return operations
            else:
                logger.warning(f"LLM returned non-list response: {operations}")
                return ["list_public_jobs"]
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response for todo list: {e}")
            return get_mock_todo_list_creation(state)

    except LLMServiceError as e:
        logger.error(f"LLMServiceError in create_todo_list: {e}")
        return get_mock_todo_list_creation(state)
    except Exception as e:
        logger.error(f"Unexpected error in create_todo_list: {e}")
        return get_mock_todo_list_creation(state)


async def extract_api_operations(text_or_state) -> List[str]:
    """Extract API operations from text or state using LLM.

    Args:
        text_or_state: Text string or GraphState to extract operations from

    Returns:
        List of API operations
    """
    if should_use_mocks():
        # Handle both string and state inputs
        if isinstance(text_or_state, dict):
            # Extract text from state
            if text_or_state.get("messages"):
                last_message = text_or_state["messages"][-1]
                content = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )
                if isinstance(content, list):
                    text = ""
                    for item in content:
                        if isinstance(item, str):
                            text = item
                            break
                else:
                    text = content or ""
            else:
                text = ""
        else:
            text = text_or_state
        return get_mock_api_operations_extraction(text)

    try:
        service = get_llm_service()

        prompt = f"""You are a Supervisor agent that extracts API operations from user text.

User text: "{text}"

Available API operations:
- list_public_jobs: List all available jobs
- run_job:job_name=<name>: Run a specific job (e.g., data_processing, image_analysis, report_generation)
- get_job_results:job_id=<id>: Get results for a specific job
- check_system_status: Check overall system status

Extract the API operations mentioned in the text. Return a JSON array of operation strings.

Examples:
- "List all jobs" → ["list_public_jobs"]
- "Run data processing job" → ["run_job:job_name=data_processing"]
- "Get results for job_123" → ["get_job_results:job_id=job_123"]
- "Check system status" → ["check_system_status"]

Respond with ONLY a JSON array - no explanation needed."""

        response_content = service.generate_with_system_prompt(
            AgentType.SUPERVISOR, prompt, get_agent_prompt("supervisor")
        )

        # Parse JSON response
        try:
            operations = json.loads(response_content.strip())
            if isinstance(operations, list):
                return operations
            else:
                logger.warning(f"LLM returned non-list response: {operations}")
                return ["list_public_jobs"]
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response for API operations: {e}")
            return get_mock_api_operations_extraction(text)

    except LLMServiceError as e:
        logger.error(f"LLMServiceError in extract_api_operations: {e}")
        return get_mock_api_operations_extraction(text)
    except Exception as e:
        logger.error(f"Unexpected error in extract_api_operations: {e}")
        return get_mock_api_operations_extraction(text)


# Legacy functions for backward compatibility
def extract_job_name(text: str) -> str:
    """Extract job name from text (legacy function).

    Args:
        text: Text to extract job name from

    Returns:
        Extracted job name
    """
    text_lower = text.lower()

    # Common job names
    if "data" in text_lower and "process" in text_lower:
        return "data_processing"
    elif "image" in text_lower and "analysis" in text_lower:
        return "image_analysis"
    elif "report" in text_lower and "generation" in text_lower:
        return "report_generation"
    elif "validation" in text_lower:
        return "data_validation"
    else:
        return "data_processing"  # Default


def extract_job_id(text: str) -> Optional[str]:
    """Extract job ID from text (legacy function).

    Args:
        text: Text to extract job ID from

    Returns:
        Extracted job ID or None
    """
    import re

    # Look for job_XXX pattern
    job_id_match = re.search(r"job_(\d{3})", text.lower())
    if job_id_match:
        return f"job_{job_id_match.group(1)}"
    return None
