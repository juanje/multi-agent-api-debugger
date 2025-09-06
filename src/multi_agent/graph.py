"""
LangGraph construction and configuration for the new multi-agent architecture.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import GraphState
from .supervisor import supervisor_node
from .api_operator import api_operator_node
from .debugger import debugger_node
from .knowledge_assistant import knowledge_assistant_node
from .response_synthesizer import response_synthesizer_node


def create_graph():
    """Creates and configures the main multi-agent system graph."""
    # Create the main graph
    graph = StateGraph(GraphState)

    # Add all agent nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("api_operator", api_operator_node)
    graph.add_node("debugger", debugger_node)
    graph.add_node("knowledge_assistant", knowledge_assistant_node)
    graph.add_node("response_synthesizer", response_synthesizer_node)

    # Define the flow
    graph.add_edge(START, "supervisor")

    # Add conditional edge from supervisor
    def should_continue(state: GraphState) -> str:
        """Determine if we should continue or end."""
        route = state.get("route", "done")
        if route == "done":
            return "response_synthesizer"
        elif route in [
            "api_operator",
            "debugger",
            "knowledge_assistant",
            "response_synthesizer",
        ]:
            return route
        else:
            return "response_synthesizer"

    graph.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "api_operator": "api_operator",
            "debugger": "debugger",
            "knowledge_assistant": "knowledge_assistant",
            "response_synthesizer": "response_synthesizer",
        },
    )

    # All agents go directly to response synthesizer
    graph.add_edge("api_operator", "response_synthesizer")
    graph.add_edge("debugger", "response_synthesizer")
    graph.add_edge("knowledge_assistant", "response_synthesizer")
    graph.add_edge("response_synthesizer", END)  # Response synthesizer ends the flow

    # Compile with memory
    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)

    return app


# Global graph instance
app = create_graph()
