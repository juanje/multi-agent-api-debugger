"""
Unit tests for supervisor.py
"""

from langchain_core.messages import HumanMessage
from multi_agent.supervisor import supervisor_node


class TestSupervisorNode:
    """Tests for the supervisor_node function."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = supervisor_node(state)
        assert result == state

    def test_sequential_route_continuation(self):
        """Test sequential route continuation."""
        state = {
            "messages": [{"content": "test"}],
            "route": "sequential",
            "steps": [{"type": "calc", "operation": "2+3", "step": 1}],
            "current_step": 0,
            "results": [],
        }
        result = supervisor_node(state)
        assert result.goto == "sequential_executor"

    def test_sequential_route_completion(self):
        """Test sequential route completion."""
        state = {
            "messages": [{"content": "test"}],
            "route": "sequential",
            "steps": [{"type": "calc", "operation": "2+3", "step": 1}],
            "current_step": 1,  # Greater than len(steps)
            "results": ["Result 1", "Final result"],
        }
        result = supervisor_node(state)
        assert result.goto == "finish"
        assert "Final result" in result.update["messages"][-1].content

    def test_sequential_operations_detection(self):
        """Test sequential operations detection."""
        state = {
            "messages": [HumanMessage(content="sum of 3x8 and 129/3")],
        }
        result = supervisor_node(state)
        assert result.goto == "sequential_executor"
        assert result.update["route"] == "sequential"
        assert result.update["current_step"] == 0
        assert result.update["results"] == []

    def test_parallel_operations_detection(self):
        """Test parallel operations detection."""
        state = {
            "messages": [HumanMessage(content="What is 2+3 and 8/2?")],
        }
        result = supervisor_node(state)
        assert result.goto == "parallel_executor"
        assert result.update["route"] == "parallel"
        assert "2+3" in result.update["operations"]
        assert "8/2" in result.update["operations"]

    def test_tools_route_calculation(self):
        """Test routing to tools for calculations."""
        state = {
            "messages": [HumanMessage(content="What is 12*7?")],
        }
        result = supervisor_node(state)
        assert result.goto == "agent_tools"
        assert result.update["route"] == "tools"

    def test_tools_route_time(self):
        """Test routing to tools for time."""
        state = {
            "messages": [HumanMessage(content="What time is it?")],
        }
        result = supervisor_node(state)
        assert result.goto == "agent_tools"
        assert result.update["route"] == "tools"

    def test_rag_route_document(self):
        """Test routing to RAG for documents."""
        state = {
            "messages": [HumanMessage(content="summarize the manual")],
        }
        result = supervisor_node(state)
        assert result.goto == "agent_rag"
        assert result.update["route"] == "rag"

    def test_rag_route_question(self):
        """Test routing to RAG for questions."""
        state = {
            "messages": [HumanMessage(content="Can you help me?")],
        }
        result = supervisor_node(state)
        assert result.goto == "agent_rag"
        assert result.update["route"] == "rag"

    def test_tools_route_ambiguous(self):
        """Test routing to tools for ambiguous input without '?'."""
        state = {
            "messages": [HumanMessage(content="hello")],
        }
        result = supervisor_node(state)
        assert result.goto == "agent_tools"
        assert result.update["route"] == "tools"

    def test_rag_route_ambiguous_with_question(self):
        """Test routing to RAG for ambiguous input with '?'."""
        state = {
            "messages": [HumanMessage(content="hello?")],
        }
        result = supervisor_node(state)
        assert result.goto == "agent_rag"
        assert result.update["route"] == "rag"

    def test_message_content_handling_list(self):
        """Test handling message content as list."""
        state = {
            "messages": [HumanMessage(content=["text", "additional"])],
        }
        result = supervisor_node(state)
        # Should process the first element of the list
        assert result.goto in ["agent_tools", "agent_rag"]

    def test_message_content_handling_none(self):
        """Test handling message content None."""
        # Create a message with empty content instead of None
        state = {
            "messages": [HumanMessage(content="")],
        }
        result = supervisor_node(state)
        # Should handle empty content correctly
        assert result.goto in ["agent_tools", "agent_rag"]
