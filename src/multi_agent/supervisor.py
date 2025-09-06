"""
Intelligent supervisor that decides the flow of operations.
"""

from langgraph.types import Command
from langchain_core.messages import AIMessage

from .state import ChatState
from .tools import (
    ROUTE_TOOLS_RE,
    ROUTE_RAG_RE,
    detect_parallel_operations,
    detect_sequential_operations,
)


def supervisor_node(state: ChatState):
    """Intelligent supervisor that detects parallel and sequential patterns."""
    if not state["messages"]:
        return state

    # Check if we're in the middle of a sequential operation
    if state.get("route") == "sequential":
        steps = state.get("steps") or []
        current_step = state.get("current_step") or 0

        if current_step < len(steps):
            # There are more steps, continue with the next one
            return Command(goto="sequential_executor")
        else:
            # No more steps, show final result and terminate
            results = state.get("results") or []
            final_result = results[-1] if results else "No operations completed"

            new_state = state.copy()
            msgs = list(new_state["messages"])
            msgs.append(AIMessage(content=f"🤖 Assistant: {final_result}"))
            new_state["messages"] = msgs
            new_state["route"] = "done"

            return Command(goto="finish", update=new_state)

    last_user = state["messages"][-1]
    content = last_user.content
    if isinstance(content, list):
        # If it's a list, take the first string element
        text = ""
        for item in content:
            if isinstance(item, str):
                text = item
                break
    else:
        text = content or ""
    text = text.lower()

    # Detect sequential operations FIRST (more specific)
    sequential_steps = detect_sequential_operations(text)
    if sequential_steps:
        return Command(
            goto="sequential_executor",
            update={
                "route": "sequential",
                "steps": sequential_steps,
                "current_step": 0,
                "results": [],
            },
        )

    # Detect parallel operations
    parallel_ops = detect_parallel_operations(text)
    if parallel_ops:
        return Command(
            goto="parallel_executor",
            update={"route": "parallel", "operations": parallel_ops, "results": []},
        )

    # Original logic for simple operations
    if ROUTE_TOOLS_RE.search(text) or ("time" in text):
        route = "tools"
    elif ROUTE_RAG_RE.search(text):
        route = "rag"
    else:
        route = "rag" if "?" in text else "tools"

    target = "agent_tools" if route == "tools" else "agent_rag"
    return Command(goto=target, update={"route": route})
