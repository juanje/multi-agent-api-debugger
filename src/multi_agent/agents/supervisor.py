"""
Supervisor agent for orchestrating the multi-agent workflow.

This agent is responsible for:
- Analyzing user requests
- Determining the next agent to invoke
- Managing the overall workflow
- Coordinating between specialized agents
"""

from __future__ import annotations
from typing import Dict, Any
from langchain_core.messages import AIMessage
from ..graph.state import GraphState
from ..graph.planning import create_comprehensive_todo_list, get_next_task
from ..graph.routing import determine_route


class Supervisor:
    """Supervisor agent for orchestrating the multi-agent workflow."""

    def __init__(self):
        """Initialize the Supervisor."""
        pass

    async def analyze_request(self, state: GraphState) -> Dict[str, Any]:
        """Analyze the user request and determine next steps using LLM."""
        if not state.get("messages"):
            return {"route": "done", "next_agent": None, "todo_list": []}

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

        try:
            # Determine route using LLM (now async)
            route = await determine_route(state)
            import logging

            logger = logging.getLogger(__name__)
            logger.info(f"Route determination result: {route}")

            # Create comprehensive todo list using LLM
            todo_list = await create_comprehensive_todo_list(state)

            # Determine next agent and instruction
            if route == "done":
                next_agent = None
                instruction = None
            else:
                next_agent = route
                # Get next task for instruction
                try:
                    next_task = await get_next_task(todo_list)
                    if next_task:
                        instruction = f"Execute: {next_task['description']}"
                    else:
                        instruction = f"Process request using {route}"
                except Exception as task_error:
                    # If get_next_task fails, use a simple instruction
                    import logging

                    logging.warning(f"get_next_task failed: {task_error}")
                    instruction = f"Process request using {route}"

            return {
                "route": route,
                "next_agent": next_agent,
                "todo_list": todo_list,
                "instruction": instruction,
            }
        except Exception as e:
            # Fallback to keyword-based routing if LLM calls fail
            import logging

            logging.warning(f"LLM calls failed, using fallback: {e}")

            text_lower = text.lower()
            if any(
                word in text_lower
                for word in ["debug", "depura", "error", "fail", "investigate"]
            ):
                route = "debugger"
                next_agent = "debugger"
                # Extract job ID if present
                if "job_" in text_lower:
                    import re

                    job_match = re.search(r"job_(\d{3})", text_lower)
                    if job_match:
                        job_id = f"job_{job_match.group(1)}"
                        instruction = f"Debug {job_id} - analyze logs and provide root cause analysis"
                    else:
                        instruction = "Debug the issue - analyze logs and provide root cause analysis"
                else:
                    instruction = (
                        "Debug the issue - analyze logs and provide root cause analysis"
                    )
            elif any(
                word in text_lower for word in ["list", "show", "get", "run", "execute"]
            ):
                route = "api_operator"
                next_agent = "api_operator"
                if "list" in text_lower or "show" in text_lower:
                    instruction = "List all available jobs"
                elif "run" in text_lower:
                    instruction = "Execute the specified job"
                elif "get" in text_lower:
                    instruction = "Get job results"
                else:
                    instruction = "Execute API operation"
            elif any(
                word in text_lower for word in ["what", "how", "explain", "help", "?"]
            ):
                route = "knowledge_assistant"
                next_agent = "knowledge_assistant"
                instruction = "Provide knowledge assistance"
            else:
                route = "api_operator"
                next_agent = "api_operator"
                instruction = "Process request"

            return {
                "route": route,
                "next_agent": next_agent,
                "todo_list": [{"description": instruction, "status": "pending"}],
                "instruction": instruction,
            }


async def supervisor_node(state: GraphState) -> GraphState:
    """Supervisor node that analyzes requests and determines next steps."""
    if not state["messages"]:
        return state

    supervisor = Supervisor()
    analysis = await supervisor.analyze_request(state)

    new_state = state.copy()
    msgs = list(new_state["messages"])

    # Store the initial user request as goal if not already set
    if not new_state.get("goal") and state["messages"]:
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
        new_state["goal"] = text

    # Update state with analysis results
    new_state["route"] = analysis["route"]
    new_state["next_agent"] = analysis["next_agent"]

    if analysis["todo_list"]:
        new_state["todo_list"] = analysis["todo_list"]

    # Add supervisor message
    if analysis["route"] == "done":
        msgs.append(AIMessage(content="✅ Supervisor: Workflow completed"))
        new_state["route"] = "done"
    else:
        next_agent = analysis["next_agent"]
        instruction = analysis.get("instruction", "Process request")
        msgs.append(
            AIMessage(content=f"🎯 Supervisor: Routing to {next_agent} - {instruction}")
        )

    new_state["messages"] = msgs
    return new_state
