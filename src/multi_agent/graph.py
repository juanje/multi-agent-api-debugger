"""
LangGraph construction and configuration.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from .state import ChatState
from .supervisor import supervisor_node
from .agent_tools import agent_tools_node
from .agent_rag import agent_rag_node
from .executors import parallel_executor_node, sequential_executor_node, finish_node


def create_graph():
    """Creates and configures the main multi-agent system graph."""
    # Parent graph
    supergraph = StateGraph(ChatState)

    # Add main nodes
    supergraph.add_node("supervisor", supervisor_node)
    supergraph.add_node("parallel_executor", parallel_executor_node)
    supergraph.add_node("sequential_executor", sequential_executor_node)
    supergraph.add_node("finish", finish_node)

    # Subgraph: Tools
    sub_tools = StateGraph(ChatState)
    sub_tools.add_node("tools_run", agent_tools_node)
    sub_tools.add_edge(START, "tools_run")
    sub_tools.add_edge("tools_run", END)  # Ends at END, returns control to parent
    agent_tools = sub_tools.compile()

    # Subgraph: RAG
    sub_rag = StateGraph(ChatState)
    sub_rag.add_node("rag_run", agent_rag_node)
    sub_rag.add_edge(START, "rag_run")
    sub_rag.add_edge("rag_run", END)
    agent_rag = sub_rag.compile()

    # Register subgraphs in parent graph
    supergraph.add_node("agent_tools", agent_tools)
    supergraph.add_node("agent_rag", agent_rag)

    # Parent graph flow
    supergraph.add_edge(START, "supervisor")
    supergraph.add_edge("agent_tools", "finish")
    supergraph.add_edge("agent_rag", "finish")
    supergraph.add_edge("parallel_executor", "finish")
    supergraph.add_edge("sequential_executor", "supervisor")  # Returns to supervisor
    supergraph.add_edge("finish", END)

    # Compile with thread memory (optional)
    memory = MemorySaver()
    app = supergraph.compile(checkpointer=memory)

    return app


# Global graph instance
app = create_graph()
