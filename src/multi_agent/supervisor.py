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
from multi_agent.state import GraphState
from multi_agent.mocks.planning import create_comprehensive_todo_list, get_next_task
from multi_agent.mocks.intelligence import (
    analyze_user_intent,
)


class Supervisor:
    """Supervisor agent for orchestrating the multi-agent workflow."""

    def __init__(self):
        """Initialize the Supervisor."""
        pass

    def analyze_request(self, state: GraphState) -> Dict[str, Any]:
        """Analyze the user request and determine next steps."""
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

        # Analyze user intent
        intent = analyze_user_intent(text)

        # Check if we already have a final response
        if state.get("final_response"):
            route = "done"
            next_agent = None
            instruction = None
            todo_list = []
        else:
            # Create comprehensive todo list for all workflows
            todo_list = create_comprehensive_todo_list(state)

            # Determine route based on next task
            next_task = get_next_task(todo_list)
            if next_task:
                route = next_task["agent"]
                next_agent = route
                instruction = f"Execute: {next_task['description']}"
            else:
                # No pending tasks, check if we need to synthesize response
                if (
                    state.get("results")
                    or state.get("root_cause_analysis")
                    or state.get("knowledge_summary")
                ):
                    route = "response_synthesizer"
                    next_agent = "response_synthesizer"
                    instruction = "Synthesize final response"
                else:
                    route = "done"
                    next_agent = None
                    instruction = None

        return {
            "route": route,
            "next_agent": next_agent,
            "todo_list": todo_list,
            "intent": intent,
            "instruction": instruction,
        }


def supervisor_node(state: GraphState) -> GraphState:
    """Supervisor node that analyzes requests and determines next steps."""
    if not state["messages"]:
        return state

    supervisor = Supervisor()
    analysis = supervisor.analyze_request(state)

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
