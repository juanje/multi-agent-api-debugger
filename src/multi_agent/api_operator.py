"""
API Operator agent for handling external API interactions.

This agent is responsible for all API calls and data retrieval operations.
"""

from __future__ import annotations
from typing import Dict, Any
from langchain_core.messages import AIMessage
from multi_agent.state import GraphState
from multi_agent.mocks.data import API_JOBS, API_SYSTEM_STATUS, get_job_result
from multi_agent.mocks.planning import (
    get_next_task,
    mark_task_completed,
    mark_task_failed,
)


class APIOperator:
    """API Operator for handling external API interactions."""

    def __init__(self):
        """Initialize the API Operator."""
        pass

    def list_public_jobs(self) -> Dict[str, Any]:
        """List all available public jobs."""
        return {"jobs": API_JOBS}

    def run_job(self, job_name: str) -> Dict[str, Any]:
        """Run a specific job."""
        job_id = f"job_{len(API_JOBS) + 1:03d}"
        return {
            "job_id": job_id,
            "job_name": job_name,
            "status": "queued",
            "message": f"Job '{job_name}' queued for execution",
        }

    def get_job_results(self, job_id: str) -> Dict[str, Any]:
        """Get results for a specific job."""
        return get_job_result(job_id)

    def check_system_status(self) -> Dict[str, Any]:
        """Check overall system status."""
        return API_SYSTEM_STATUS

    def execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific tool with given parameters."""
        if tool_name == "list_public_jobs":
            return self.list_public_jobs()
        elif tool_name == "run_job":
            return self.run_job(params.get("job_name", "unknown"))
        elif tool_name == "get_job_results":
            return self.get_job_results(params.get("job_id", "unknown"))
        elif tool_name == "check_system_status":
            return self.check_system_status()
        else:
            return {"error": f"Unknown tool: {tool_name}"}


def api_operator_node(state: GraphState) -> GraphState:
    """API Operator node that executes API calls based on task instructions."""
    if not state["messages"]:
        return state

    todo_list = state.get("todo_list", [])
    if not todo_list:
        return state

    # Get the next task for API operator
    next_task = get_next_task(todo_list)
    if not next_task or next_task["agent"] != "api_operator":
        return state

    new_state = state.copy()
    msgs = list(new_state["messages"])
    current_results = new_state.get("results") or {}

    operator = APIOperator()

    # Execute the task
    operation = next_task["parameters"].get("operation", "list_public_jobs")
    task_params = {k: v for k, v in next_task["parameters"].items() if k != "operation"}

    result = operator.execute_tool(operation, task_params)

    msgs.append(AIMessage(content=f"🔧 API Operator executing: {operation}"))
    msgs.append(AIMessage(content=f"📋 Task: {next_task['description']}"))

    # Store results in state
    current_results[operation] = result
    new_state["results"] = current_results

    # Update task status
    if "error" in result:
        new_state["error_info"] = result["error"]
        msgs.append(AIMessage(content=f"❌ API Error: {result['error']}"))
        todo_list = mark_task_failed(todo_list, next_task["id"], result["error"])
    else:
        msgs.append(AIMessage(content=f"✅ API Success: {result}"))
        todo_list = mark_task_completed(todo_list, next_task["id"], result)

    new_state["messages"] = msgs
    new_state["todo_list"] = todo_list
    return new_state
