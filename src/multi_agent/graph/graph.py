"""
LangGraph construction and configuration for the new multi-agent architecture.
"""

from typing import Union
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GraphState
from ..agents import (
    supervisor_node,
    api_operator_node,
    debugger_node,
    knowledge_assistant_node,
    response_synthesizer_node,
)


# All nodes are now asynchronous


def create_graph():
    """Creates and configures the main multi-agent system graph."""
    # Create the state graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("api_operator", api_operator_node)
    workflow.add_node("debugger", debugger_node)
    workflow.add_node("knowledge_assistant", knowledge_assistant_node)
    workflow.add_node("response_synthesizer", response_synthesizer_node)

    # Define the routing logic
    def route_decision(state: GraphState) -> Union[str, type(END)]:
        """Route to the next node based on the current state."""
        route = state.get("route", "supervisor")

        # If we have an error, always go to response_synthesizer
        if state.get("error"):
            return "response_synthesizer"

        # If route is "done", end the workflow
        if route == "done":
            return END

        return route

    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "api_operator": "api_operator",
            "debugger": "debugger",
            "knowledge_assistant": "knowledge_assistant",
            "response_synthesizer": "response_synthesizer",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "api_operator",
        route_decision,
        {
            "supervisor": "supervisor",
            "response_synthesizer": "response_synthesizer",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "debugger",
        route_decision,
        {
            "supervisor": "supervisor",
            "response_synthesizer": "response_synthesizer",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "knowledge_assistant",
        route_decision,
        {
            "supervisor": "supervisor",
            "response_synthesizer": "response_synthesizer",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "response_synthesizer",
        route_decision,
        {
            "supervisor": "supervisor",
            END: END,
        },
    )

    # Compile the graph
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def get_graph():
    """Get the compiled graph instance."""
    return create_graph()
