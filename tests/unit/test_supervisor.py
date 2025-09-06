"""
Unit tests for the Supervisor agent.
"""

from langchain_core.messages import HumanMessage
from multi_agent.supervisor import Supervisor, supervisor_node
from multi_agent.state import GraphState


class TestSupervisor:
    """Test cases for the Supervisor class."""

    def test_init(self):
        """Test supervisor initialization."""
        supervisor = Supervisor()
        assert supervisor is not None

    def test_analyze_request_empty_messages(self):
        """Test analyze_request with empty messages."""
        supervisor = Supervisor()
        state = GraphState(messages=[])
        result = supervisor.analyze_request(state)

        assert result["route"] == "done"
        assert result["next_agent"] is None
        assert result["todo_list"] == []

    def test_analyze_request_with_message(self):
        """Test analyze_request with a message."""
        supervisor = Supervisor()
        state = GraphState(messages=[HumanMessage(content="list all jobs")])
        result = supervisor.analyze_request(state)

        assert "route" in result
        assert "next_agent" in result
        assert "todo_list" in result
        assert "intent" in result
        assert "instruction" in result

    def test_analyze_request_list_jobs(self):
        """Test analyze_request for list jobs command."""
        supervisor = Supervisor()
        state = GraphState(messages=[HumanMessage(content="list all jobs")])
        result = supervisor.analyze_request(state)

        assert result["route"] == "api_operator"
        assert result["next_agent"] == "api_operator"
        assert len(result["todo_list"]) > 0
        assert result["todo_list"][0]["agent"] == "api_operator"

    def test_analyze_request_knowledge_query(self):
        """Test analyze_request for knowledge query."""
        supervisor = Supervisor()
        state = GraphState(messages=[HumanMessage(content="what are jobs?")])
        result = supervisor.analyze_request(state)

        assert result["route"] == "knowledge_assistant"
        assert result["next_agent"] == "knowledge_assistant"
        assert len(result["todo_list"]) > 0
        assert result["todo_list"][0]["agent"] == "knowledge_assistant"

    def test_analyze_request_debug_command(self):
        """Test analyze_request for debug command."""
        supervisor = Supervisor()
        state = GraphState(messages=[HumanMessage(content="debug job_003")])
        result = supervisor.analyze_request(state)

        assert result["route"] == "debugger"
        assert result["next_agent"] == "debugger"
        assert len(result["todo_list"]) > 0
        assert result["todo_list"][0]["agent"] == "debugger"

    def test_analyze_request_with_list_content(self):
        """Test analyze_request with list content in message."""
        supervisor = Supervisor()
        # Create a message with list content (simulating complex message format)
        message = HumanMessage(content=["list", "all", "jobs"])
        state = GraphState(messages=[message])
        result = supervisor.analyze_request(state)

        assert "route" in result
        assert "next_agent" in result

    def test_analyze_request_with_none_content(self):
        """Test analyze_request with None content."""
        supervisor = Supervisor()
        # Use empty string instead of None since HumanMessage doesn't accept None
        message = HumanMessage(content="")
        state = GraphState(messages=[message])
        result = supervisor.analyze_request(state)

        assert "route" in result
        assert "next_agent" in result


class TestSupervisorNode:
    """Test cases for the supervisor_node function."""

    def test_empty_messages(self):
        """Test supervisor_node with empty messages."""
        state = GraphState(messages=[])
        result = supervisor_node(state)
        assert result == state

    def test_with_message(self):
        """Test supervisor_node with a message."""
        state = GraphState(messages=[HumanMessage(content="list all jobs")])
        result = supervisor_node(state)

        assert "messages" in result
        assert len(result["messages"]) > len(state["messages"])
        assert "route" in result
        assert "next_agent" in result
        assert "todo_list" in result

    def test_goal_setting(self):
        """Test that goal is set from user message."""
        state = GraphState(messages=[HumanMessage(content="test message")])
        result = supervisor_node(state)

        assert result.get("goal") == "test message"

    def test_goal_not_overwritten(self):
        """Test that existing goal is not overwritten."""
        state = GraphState(
            messages=[HumanMessage(content="new message")], goal="existing goal"
        )
        result = supervisor_node(state)

        assert result.get("goal") == "existing goal"

    def test_route_done_workflow_completed(self):
        """Test supervisor message when workflow is completed."""
        # Create a state that would result in route="done" by having final_response
        state = GraphState(
            messages=[HumanMessage(content="test")], final_response="Test response"
        )
        result = supervisor_node(state)

        # Check that we have a completion message
        messages = result["messages"]
        completion_messages = [
            msg for msg in messages if "Workflow completed" in str(msg.content)
        ]
        assert len(completion_messages) > 0

    def test_routing_message_generated(self):
        """Test that routing message is generated for non-done routes."""
        state = GraphState(messages=[HumanMessage(content="list all jobs")])
        result = supervisor_node(state)

        # Check that we have a routing message
        messages = result["messages"]
        routing_messages = [msg for msg in messages if "Routing to" in str(msg.content)]
        assert len(routing_messages) > 0

    def test_todo_list_updated(self):
        """Test that todo_list is updated in state."""
        state = GraphState(messages=[HumanMessage(content="list all jobs")])
        result = supervisor_node(state)

        assert "todo_list" in result
        assert isinstance(result["todo_list"], list)
        if result["todo_list"]:
            assert "id" in result["todo_list"][0]
            assert "description" in result["todo_list"][0]
            assert "agent" in result["todo_list"][0]

    def test_state_copy_preserved(self):
        """Test that original state is preserved in copy."""
        original_state = GraphState(
            messages=[HumanMessage(content="test")],
            goal="test goal",
            results={"test": "data"},
        )
        result = supervisor_node(original_state)

        # Check that original fields are preserved
        assert result.get("goal") == "test goal"
        assert result.get("results") == {"test": "data"}
        assert len(result["messages"]) >= len(original_state["messages"])
