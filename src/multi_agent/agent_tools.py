"""
Tools agent (calculator, time, etc.).
"""

from langchain_core.messages import AIMessage

from .state import ChatState
from .tools import CALC_RE, tool_calculator, tool_time


def agent_tools_node(state: ChatState) -> ChatState:
    """Tools agent node."""
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

    if CALC_RE.search(user_msg):
        out = tool_calculator(user_msg)
    elif "time" in user_msg.lower():
        out = tool_time(user_msg)
    else:
        out = "I didn't identify a specific tool. (mock tools)"

    new_state = state.copy()
    msgs = list(new_state["messages"])  # List[BaseMessage]

    # Add intermediate process message
    msgs.append(AIMessage(content="🔧 Tools agent working..."))
    msgs.append(AIMessage(content=out))
    new_state["messages"] = msgs
    return new_state
