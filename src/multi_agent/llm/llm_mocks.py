"""
Mock responses for LLM testing and development.

This module provides mock responses for LLM functions to enable fast testing
without requiring actual LLM calls. This is especially useful for:
- Unit testing
- Development and debugging
- CI/CD pipelines
- Performance testing
"""

from __future__ import annotations
import os
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, MagicMock


class LLMMockResponses:
    """Centralized mock responses for LLM functions."""

    # Mock responses for routing
    ROUTE_RESPONSES = {
        "list": {"next_agent": "api_operator", "reasoning": "User wants to list items"},
        "run": {
            "next_agent": "api_operator",
            "reasoning": "User wants to execute something",
        },
        "debug": {
            "next_agent": "debugger",
            "reasoning": "User wants to debug an issue",
        },
        "error": {
            "next_agent": "debugger",
            "reasoning": "Error mentioned in the request",
        },
        "what": {
            "next_agent": "knowledge_assistant",
            "reasoning": "User is asking a question",
        },
        "how": {
            "next_agent": "knowledge_assistant",
            "reasoning": "User is asking for information",
        },
        "help": {
            "next_agent": "knowledge_assistant",
            "reasoning": "User needs assistance",
        },
    }

    # Mock task creation templates
    TASK_TEMPLATES = {
        "api_operator": [
            {
                "id": "task_api_1",
                "description": "Execute the requested API operation",
                "agent": "api_operator",
                "status": "pending",
                "parameters": {"operation": "list_public_jobs"},
            }
        ],
        "debugger": [
            {
                "id": "task_debug_1",
                "description": "Analyze the error and provide root cause analysis",
                "agent": "debugger",
                "status": "pending",
                "parameters": {"error_type": "general"},
            }
        ],
        "knowledge_assistant": [
            {
                "id": "task_knowledge_1",
                "description": "Answer the user's question",
                "agent": "knowledge_assistant",
                "status": "pending",
                "parameters": {"query_type": "general"},
            }
        ],
    }


def should_use_mocks() -> bool:
    """Check if LLM mocking should be used.

    Returns:
        True if mocking should be used, False otherwise
    """
    return os.getenv("USE_LLM_MOCKS", "false").lower() == "true"


def get_mock_response_for_function(function_name: str, *args, **kwargs) -> Any:
    """Get mock response for a specific function.

    Args:
        function_name: Name of the function to mock
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Mock response
    """
    # All intelligence functions have been removed as they are no longer used
    return "Mock response"


def get_mock_route_determination(state: Union[dict, Any]) -> str:
    """Mock route determination based on state.

    Args:
        state: Current graph state

    Returns:
        Next agent route
    """
    # Convert state to dict if it's a GraphState object
    if hasattr(state, "get"):
        state_dict = state
    else:
        state_dict = dict(state) if state else {}

    # Handle error states
    if state_dict.get("error_info") or state_dict.get("error"):
        return "debugger"

    # Check if workflow is complete
    todo_list = state_dict.get("todo_list", [])
    if todo_list:
        # Check if all tasks are completed
        completed_tasks = [t for t in todo_list if t.get("status") == "completed"]
        if len(completed_tasks) == len(todo_list):
            return "response_synthesizer"

        # Find pending tasks
        pending_tasks = [t for t in todo_list if t.get("status") == "pending"]
        if pending_tasks:
            task = pending_tasks[0]
            return task.get("agent", "api_operator")

    # If there's a final response, we're done
    if state_dict.get("final_response"):
        return "done"

    # Default routing based on message content
    if not state_dict.get("messages"):
        return "done"

    last_message = state_dict["messages"][-1]
    content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    if isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, str):
                text += item + " "
        content = text.strip()

    content_lower = content.lower()

    # Match against known patterns
    for pattern, response in LLMMockResponses.ROUTE_RESPONSES.items():
        if pattern in content_lower:
            return response["next_agent"]

    # Default to api_operator for most requests
    return "api_operator"


def get_mock_todo_list_creation(state: Union[dict, Any]) -> list:
    """Mock todo list creation based on state.

    Args:
        state: Current graph state

    Returns:
        List of mock tasks
    """
    # Convert state to dict if it's a GraphState object
    if hasattr(state, "get"):
        state_dict = state
    else:
        state_dict = dict(state) if state else {}

    if not state_dict.get("messages"):
        return []

    last_message = state_dict["messages"][-1]
    content = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    if isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, str):
                text += item + " "
        content = text.strip()

    content_lower = content.lower()

    # Determine task type and return appropriate template
    if any(word in content_lower for word in ["debug", "error", "fail", "job_"]):
        return LLMMockResponses.TASK_TEMPLATES["debugger"].copy()
    elif any(word in content_lower for word in ["what", "how", "help", "explain"]):
        return LLMMockResponses.TASK_TEMPLATES["knowledge_assistant"].copy()
    else:
        return LLMMockResponses.TASK_TEMPLATES["api_operator"].copy()


def get_mock_api_operations_extraction(text: str) -> list:
    """Mock API operations extraction from text.

    Args:
        text: Text to extract operations from

    Returns:
        List of mock API operations
    """
    text_lower = text.lower()
    operations = []

    if "list" in text_lower and "job" in text_lower:
        operations.append({"operation": "list_public_jobs", "parameters": {}})
    elif "run" in text_lower and "job" in text_lower:
        # Extract job name from text if possible
        job_name = "data_processing"  # Default for tests
        if "data_processing" in text_lower:
            job_name = "data_processing"
        operations.append(
            {"operation": "run_job", "parameters": {"job_name": job_name}}
        )
    elif "status" in text_lower:
        operations.append({"operation": "check_system_status", "parameters": {}})
    elif "result" in text_lower:
        # Extract job ID from text if possible
        job_id = "job_123"  # Default for tests
        if "job_123" in text:
            job_id = "job_123"
        operations.append(
            {"operation": "get_job_results", "parameters": {"job_id": job_id}}
        )

    return operations


def get_mock_comprehensive_todo_list(state: Union[dict, Any]) -> list:
    """Mock comprehensive todo list creation.

    Args:
        state: Current graph state

    Returns:
        List of comprehensive mock tasks
    """
    return get_mock_todo_list_creation(state)


def get_mock_next_task(todo_list: list) -> Optional[dict]:
    """Mock getting the next task from todo list.

    Args:
        todo_list: List of tasks

    Returns:
        Next pending task or None
    """
    if not todo_list:
        return None

    # Find first pending task
    for task in todo_list:
        if task.get("status") == "pending":
            return task

    return None


# Legacy mock creation for backwards compatibility
def create_mock_service() -> MagicMock:
    """Create a mock LLM service for testing.

    Returns:
        Mock service object
    """
    mock_service = MagicMock()
    mock_service.generate_with_system_prompt.return_value = "Mock LLM response"
    return mock_service


def create_async_mock_service() -> AsyncMock:
    """Create an async mock LLM service for testing.

    Returns:
        Async mock service object
    """
    mock_service = AsyncMock()
    mock_service.generate_with_system_prompt.return_value = "Mock LLM response"
    return mock_service
