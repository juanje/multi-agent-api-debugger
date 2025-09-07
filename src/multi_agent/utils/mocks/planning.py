"""
Mock planning intelligence for task management and workflow orchestration.

This module contains all the planning logic that would normally be handled
by an LLM in a real implementation for task decomposition and management.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from ...graph.state import GraphState


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


def create_comprehensive_todo_list(state: GraphState) -> List[Dict[str, Any]]:
    """Create a comprehensive todo list based on user request (mock LLM planning)."""
    if not state.get("messages"):
        return []

    last_message = state["messages"][-1]
    content = last_message.content
    if isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, str):
                text = item
                break
    else:
        text = content or ""

    text_lower = text.lower()
    tasks = []

    # API Operations Planning
    if any(word in text_lower for word in ["list", "show"]) and "job" in text_lower:
        tasks.append(
            create_task(
                description="List all available jobs",
                agent="api_operator",
                priority=1,
                parameters={"operation": "list_public_jobs"},
            )
        )

    elif (
        any(word in text_lower for word in ["run", "start", "execute"])
        and "job" in text_lower
    ):
        # Extract job name
        job_name = extract_job_name_from_text(text)
        tasks.append(
            create_task(
                description=f"Execute job: {job_name}",
                agent="api_operator",
                priority=1,
                parameters={"operation": "run_job", "job_name": job_name},
            )
        )

    elif (
        any(word in text_lower for word in ["check", "monitor"])
        and "system" in text_lower
    ):
        tasks.append(
            create_task(
                description="Check system status",
                agent="api_operator",
                priority=1,
                parameters={"operation": "check_system_status"},
            )
        )

    elif any(word in text_lower for word in ["get", "fetch"]) and (
        "result" in text_lower or "status" in text_lower
    ):
        job_id = extract_job_id_from_text(text)
        if job_id:
            tasks.append(
                create_task(
                    description=f"Get results for {job_id}",
                    agent="api_operator",
                    priority=1,
                    parameters={"operation": "get_job_results", "job_id": job_id},
                )
            )

    # Knowledge Queries Planning
    elif any(word in text_lower for word in ["what", "how", "explain", "tell"]):
        tasks.append(
            create_task(
                description="Answer knowledge question",
                agent="knowledge_assistant",
                priority=1,
                parameters={"query": text},
            )
        )

    # Debugging Planning
    elif (
        any(word in text_lower for word in ["debug", "analyze", "investigate"])
        and "job" in text_lower
    ):
        job_id = extract_job_id_from_text(text)
        if job_id:
            # Create a multi-step debugging workflow
            tasks.append(
                create_task(
                    description=f"Analyze error for {job_id}",
                    agent="debugger",
                    priority=1,
                    parameters={"job_id": job_id, "error_type": "template_not_found"},
                )
            )

            # Add follow-up task for response synthesis
            tasks.append(
                create_task(
                    description="Synthesize debugging response",
                    agent="response_synthesizer",
                    priority=2,
                    dependencies=[tasks[0]["id"]],
                    parameters={"response_type": "debugging"},
                )
            )

    # Default fallback
    if not tasks:
        tasks.append(
            create_task(
                description="Process user request",
                agent="api_operator",
                priority=1,
                parameters={"operation": "list_public_jobs"},
            )
        )

    return tasks


def get_next_task(todo_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Get the next task to execute based on priority and dependencies."""
    if not todo_list:
        return None

    # Filter tasks that are ready to execute (dependencies completed)
    ready_tasks = []
    for task in todo_list:
        if task["status"] == "pending":
            # Check if all dependencies are completed
            dependencies_completed = True
            for dep_id in task["dependencies"]:
                dep_task = next((t for t in todo_list if t["id"] == dep_id), None)
                if not dep_task or dep_task["status"] != "completed":
                    dependencies_completed = False
                    break

            if dependencies_completed:
                ready_tasks.append(task)

    if not ready_tasks:
        return None

    # Sort by priority (lower number = higher priority)
    ready_tasks.sort(key=lambda x: x["priority"])
    return ready_tasks[0]


def mark_task_completed(
    todo_list: List[Dict[str, Any]], task_id: str, result: Any = None
) -> List[Dict[str, Any]]:
    """Mark a task as completed and store its result."""
    for task in todo_list:
        if task["id"] == task_id:
            task["status"] = "completed"
            task["result"] = result
            break
    return todo_list


def mark_task_failed(
    todo_list: List[Dict[str, Any]], task_id: str, error: str
) -> List[Dict[str, Any]]:
    """Mark a task as failed and store the error."""
    for task in todo_list:
        if task["id"] == task_id:
            task["status"] = "failed"
            task["error"] = error
            break
    return todo_list


def get_pending_tasks(todo_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all pending tasks."""
    return [task for task in todo_list if task["status"] == "pending"]


def get_completed_tasks(todo_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all completed tasks."""
    return [task for task in todo_list if task["status"] == "completed"]


def get_failed_tasks(todo_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get all failed tasks."""
    return [task for task in todo_list if task["status"] == "failed"]


def is_workflow_complete(todo_list: List[Dict[str, Any]]) -> bool:
    """Check if all tasks in the workflow are completed or failed."""
    if not todo_list:
        return True

    return all(task["status"] in ["completed", "failed"] for task in todo_list)


def extract_job_name_from_text(text: str) -> str:
    """Extract job name from text (mock LLM extraction)."""
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
    """Extract job ID from text (mock LLM extraction)."""
    import re

    # Look for job_XXX pattern
    job_id_match = re.search(r"job_(\d{3})", text.lower())
    if job_id_match:
        return f"job_{job_id_match.group(1)}"
    return None
