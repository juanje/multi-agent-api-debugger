"""
RAG agent (Retrieval-Augmented Generation).
"""

from langchain_core.messages import AIMessage

from .state import ChatState
from .tools import mock_retrieve, mock_rag_answer


def agent_rag_node(state: ChatState) -> ChatState:
    """RAG agent node."""
    if not state["messages"]:
        return state

    last_message = state["messages"][-1]
    content = last_message.content
    if isinstance(content, list):
        # If it's a list, take the first string element
        user_msg = ""
        for item in content:
            if isinstance(item, str):
                user_msg = item
                break
    else:
        user_msg = content or ""

    docs = mock_retrieve(user_msg)
    out = mock_rag_answer(user_msg, docs)

    new_state = state.copy()
    msgs = list(new_state["messages"])

    # Add intermediate process message
    msgs.append(AIMessage(content="📚 RAG agent working..."))
    msgs.append(AIMessage(content=out))
    new_state["messages"] = msgs
    return new_state
