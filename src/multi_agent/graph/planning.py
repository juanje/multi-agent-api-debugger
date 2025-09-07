"""
Planning intelligence for task management and workflow orchestration using LLMs.

This module contains planning logic that uses LLMs to create comprehensive
task lists and manage workflow orchestration.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from .state import GraphState
from ..llm import AgentType, get_llm_service, should_use_mocks
from ..llm.llm_service import LLMServiceError
from ..llm.llm_mocks import (
    get_mock_comprehensive_todo_list,
    get_mock_next_task,
)
from ..llm.prompts import get_agent_prompt

logger = logging.getLogger(__name__)

# Global task counter for generating unique task IDs
_task_counter: List[str] = []


def create_task(
    description: str,
    agent: str,
    priority: int = 1,
    dependencies: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a new task with the given parameters."""
    global _task_counter
    task_id = f"task_{len(_task_counter) + 1:03d}"
    _task_counter.append(task_id)

    return {
        "id": task_id,
        "description": description,
        "agent": agent,
        "status": "pending",
        "priority": priority,
        "dependencies": dependencies or [],
        "parameters": parameters or {},
        "result": None,
        "error": None,
    }


async def create_comprehensive_todo_list(state: GraphState) -> List[Dict[str, Any]]:
    """Create a comprehensive todo list based on user request using LLM.

    Args:
        state: Current graph state

    Returns:
        List of tasks to be executed
    """
    if should_use_mocks():
        return get_mock_comprehensive_todo_list(state)

    try:
        # Use async LLM service
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

        prompt = f"""You are a Supervisor agent responsible for creating comprehensive task lists for multi-agent workflows.

User request: "{text}"

Available agents and their capabilities:
- api_operator: Handles API operations (list_public_jobs, run_job, get_job_results, check_system_status)
- debugger: Analyzes errors, logs, and provides root cause analysis
- knowledge_assistant: Answers questions and provides information
- response_synthesizer: Formats and presents final responses to users

Task structure:
{{
    "id": "task_001",
    "description": "Human-readable task description",
    "agent": "agent_name",
    "status": "pending",
    "priority": 1,
    "dependencies": [],
    "parameters": {{"key": "value"}},
    "result": null,
    "error": null
}}

Create a comprehensive task list that breaks down the user request into specific, actionable tasks. Consider:
1. What API operations are needed?
2. Are there any knowledge questions to answer?
3. Is debugging or error analysis required?
4. Do tasks have dependencies on each other?
5. What priority should each task have?

Return a JSON array of task objects. Be specific with parameters and consider multi-step workflows.

Examples:
- "List all jobs" → [{{"id": "task_001", "description": "List all available jobs", "agent": "api_operator", "status": "pending", "priority": 1, "dependencies": [], "parameters": {{"operation": "list_public_jobs"}}, "result": null, "error": null}}]
- "Run data processing job" → [{{"id": "task_001", "description": "Execute data processing job", "agent": "api_operator", "status": "pending", "priority": 1, "dependencies": [], "parameters": {{"operation": "run_job", "job_name": "data_processing"}}, "result": null, "error": null}}]

Respond with ONLY a JSON array - no explanation needed."""

        response_content = service.generate_with_system_prompt(
            AgentType.SUPERVISOR, prompt, get_agent_prompt("supervisor")
        )

        # Parse JSON response
        try:
            tasks = json.loads(response_content.strip())
            if isinstance(tasks, list):
                # Ensure all tasks have required fields
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    task.setdefault("id", f"task_{len(_task_counter) + 1:03d}")
                    task.setdefault("status", "pending")
                    task.setdefault("priority", 1)
                    task.setdefault("dependencies", [])
                    task.setdefault("parameters", {})
                    task.setdefault("result", None)
                    task.setdefault("error", None)
                return tasks
            else:
                logger.warning(f"LLM returned non-list response: {tasks}")
                return get_mock_comprehensive_todo_list(state)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response for todo list: {e}")
            return get_mock_comprehensive_todo_list(state)

    except LLMServiceError as e:
        logger.error(f"LLMServiceError in create_comprehensive_todo_list: {e}")
        return get_mock_comprehensive_todo_list(state)
    except Exception as e:
        logger.error(f"Unexpected error in create_comprehensive_todo_list: {e}")
        return get_mock_comprehensive_todo_list(state)


async def get_next_task(todo_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the next task to execute based on priority and dependencies using LLM.

    Args:
        todo_list: List of tasks

    Returns:
        Next task to execute or None
    """
    if should_use_mocks():
        return get_mock_next_task(todo_list)

    try:
        # Use async LLM service
        service = get_llm_service()

        # Filter pending tasks
        pending_tasks = [task for task in todo_list if task.get("status") == "pending"]
        if not pending_tasks:
            return None

        # Check dependencies for each pending task
        ready_tasks = []
        for task in pending_tasks:
            dependencies_completed = True
            for dep_id in task.get("dependencies", []):
                dep_task = next((t for t in todo_list if t.get("id") == dep_id), None)
                if not dep_task or dep_task.get("status") != "completed":
                    dependencies_completed = False
                    break

            if dependencies_completed:
                ready_tasks.append(task)

        if not ready_tasks:
            return None

        # Use LLM to select the best next task
        prompt = f"""You are a Supervisor agent responsible for selecting the next task to execute from a list of ready tasks.

Ready tasks:
{json.dumps(ready_tasks, indent=2)}

Selection criteria:
1. Priority (lower number = higher priority)
2. Task dependencies (all dependencies must be completed)
3. Logical workflow order
4. Resource efficiency

Select the most appropriate next task to execute. Consider the overall workflow and ensure efficient execution.

Return ONLY the task ID of the selected task, or "none" if no task should be executed."""

        response_content = service.generate_with_system_prompt(
            AgentType.SUPERVISOR, prompt, get_agent_prompt("supervisor")
        )

        # Parse response
        selected_task_id = response_content.strip().lower()
        if selected_task_id == "none":
            return None

        # Find the selected task
        for task in ready_tasks:
            if task.get("id") == selected_task_id:
                return task

        # Fallback to priority-based selection
        ready_tasks.sort(key=lambda x: x.get("priority", 1))
        return ready_tasks[0]

    except LLMServiceError as e:
        logger.error(f"LLMServiceError in get_next_task: {e}")
        return get_mock_next_task(todo_list)
    except Exception as e:
        logger.error(f"Unexpected error in get_next_task: {e}")
        return get_mock_next_task(todo_list)


def mark_task_completed(
    todo_list: List[Dict[str, Any]], task_id: str, result: Any = None
) -> List[Dict[str, Any]]:
    """Mark a task as completed and store its result."""
    for task in todo_list:
        if task.get("id") == task_id:
            task["status"] = "completed"
            task["result"] = result
            break
    return todo_list


def mark_task_failed(
    todo_list: List[Dict[str, Any]], task_id: str, error: str
) -> List[Dict[str, Any]]:
    """Mark a task as failed and store the error."""
    for task in todo_list:
        if task.get("id") == task_id:
            task["status"] = "failed"
            task["error"] = error
            break
    return todo_list


def get_pending_tasks(todo_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all pending tasks."""
    return [task for task in todo_list if task.get("status") == "pending"]


def get_completed_tasks(todo_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all completed tasks."""
    return [task for task in todo_list if task.get("status") == "completed"]


def get_failed_tasks(todo_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all failed tasks."""
    return [task for task in todo_list if task.get("status") == "failed"]


def is_workflow_complete(todo_list: List[Dict[str, Any]]) -> bool:
    """Check if all tasks in the workflow are completed or failed."""
    if not todo_list:
        return True

    return all(task.get("status") in ["completed", "failed"] for task in todo_list)


# Legacy functions for backward compatibility
def extract_job_name_from_text(text: str) -> str:
    """Extract job name from text (legacy function)."""
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


def extract_job_id_from_text(text: str) -> Optional[str]:
    """Extract job ID from text (legacy function)."""
    import re

    # Look for job_XXX pattern
    job_id_match = re.search(r"job_(\d{3})", text.lower())
    if job_id_match:
        return f"job_{job_id_match.group(1)}"
    return None
