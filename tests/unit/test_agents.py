"""
Unit tests for agents (agent_tools.py and agent_rag.py)
"""

from langchain_core.messages import HumanMessage
from multi_agent.agent_tools import agent_tools_node
from multi_agent.agent_rag import agent_rag_node


class TestAgentToolsNode:
    """Tests for agent_tools_node."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = agent_tools_node(state)
        assert result == state

    def test_calculation_message(self):
        """Test with calculation message."""
        state = {
            "messages": [HumanMessage(content="What is 12*7?")],
        }
        result = agent_tools_node(state)
        assert len(result["messages"]) >= 2
        assert "Tools agent working" in result["messages"][-2].content
        assert "12*7 = 84" in result["messages"][-1].content

    def test_time_message(self):
        """Test with time message."""
        state = {
            "messages": [HumanMessage(content="What time is it?")],
        }
        result = agent_tools_node(state)
        assert len(result["messages"]) >= 2
        assert "Tools agent working" in result["messages"][-2].content
        assert "Current time (mock):" in result["messages"][-1].content

    def test_unknown_message(self):
        """Test with unknown message."""
        state = {
            "messages": [HumanMessage(content="hello world")],
        }
        result = agent_tools_node(state)
        assert len(result["messages"]) >= 2
        assert "Tools agent working" in result["messages"][-2].content
        assert "mock tools" in result["messages"][-1].content

    def test_message_content_list(self):
        """Test with message content as list."""
        state = {
            "messages": [HumanMessage(content=["What is 5+3?", "additional text"])],
        }
        result = agent_tools_node(state)
        assert len(result["messages"]) >= 2
        assert "5+3 = 8" in result["messages"][-1].content

    def test_message_content_none(self):
        """Test with message content None."""
        # Create a message with empty content instead of None
        state = {
            "messages": [HumanMessage(content="")],
        }
        result = agent_tools_node(state)
        assert len(result["messages"]) >= 2
        assert "mock tools" in result["messages"][-1].content


class TestAgentRagNode:
    """Tests for agent_rag_node."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = agent_rag_node(state)
        assert result == state

    def test_document_message(self):
        """Test with document message."""
        state = {
            "messages": [HumanMessage(content="summarize the manual")],
        }
        result = agent_rag_node(state)
        assert len(result["messages"]) >= 2
        assert "RAG agent working" in result["messages"][-2].content
        assert "(RAG-mock)" in result["messages"][-1].content
        assert "summarize the manual" in result["messages"][-1].content

    def test_question_message(self):
        """Test with question message."""
        state = {
            "messages": [HumanMessage(content="Can you help me?")],
        }
        result = agent_rag_node(state)
        assert len(result["messages"]) >= 2
        assert "RAG agent working" in result["messages"][-2].content
        assert "(RAG-mock)" in result["messages"][-1].content
        assert "Can you help me?" in result["messages"][-1].content

    def test_message_content_list(self):
        """Test with message content as list."""
        state = {
            "messages": [
                HumanMessage(content=["search for information", "additional text"])
            ],
        }
        result = agent_rag_node(state)
        assert len(result["messages"]) >= 2
        assert "RAG agent working" in result["messages"][-2].content
        assert "(RAG-mock)" in result["messages"][-1].content

    def test_message_content_none(self):
        """Test with message content None."""
        # Create a message with empty content instead of None
        state = {
            "messages": [HumanMessage(content="")],
        }
        result = agent_rag_node(state)
        assert len(result["messages"]) >= 2
        assert "RAG agent working" in result["messages"][-2].content
        assert "(RAG-mock)" in result["messages"][-1].content
