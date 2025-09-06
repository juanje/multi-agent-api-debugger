"""
Unit tests for state.py
"""

from multi_agent.state import ChatState


class TestChatState:
    """Tests for the ChatState class."""

    def test_chat_state_creation(self):
        """Test ChatState creation."""
        state = ChatState(
            messages=[],
            route="tools",
            operations=["calc"],
            steps=[{"type": "calc", "operation": "2+3", "step": 1}],
            results=["5"],
            current_step=0,
            pending_tasks=["task1"],
        )

        assert state["messages"] == []
        assert state["route"] == "tools"
        assert state["operations"] == ["calc"]
        assert state["steps"] == [{"type": "calc", "operation": "2+3", "step": 1}]
        assert state["results"] == ["5"]
        assert state["current_step"] == 0
        assert state["pending_tasks"] == ["task1"]

    def test_chat_state_optional_fields(self):
        """Test ChatState with optional fields."""
        state = ChatState(
            messages=[],
            route=None,
            operations=None,
            steps=None,
            results=None,
            current_step=None,
            pending_tasks=None,
        )

        assert state["messages"] == []
        assert state["route"] is None
        assert state["operations"] is None
        assert state["steps"] is None
        assert state["results"] is None
        assert state["current_step"] is None
        assert state["pending_tasks"] is None

    def test_chat_state_inheritance(self):
        """Test that ChatState is a TypedDict."""

        # Verify that ChatState is a TypedDict
        assert hasattr(ChatState, "__annotations__")
        assert "messages" in ChatState.__annotations__
        assert "route" in ChatState.__annotations__

    def test_chat_state_route_values(self):
        """Test valid values for route."""
        valid_routes = ["tools", "rag", "parallel", "sequential", "done"]

        for route in valid_routes:
            state = ChatState(messages=[], route=route)
            assert state["route"] == route

    def test_chat_state_operations_list(self):
        """Test operations as list."""
        operations = ["calc", "time", "rag"]
        state = ChatState(messages=[], operations=operations)
        assert state["operations"] == operations

    def test_chat_state_steps_list(self):
        """Test steps as list of dictionaries."""
        steps = [
            {"type": "calc", "operation": "2+3", "step": 1},
            {"type": "time", "step": 2},
        ]
        state = ChatState(messages=[], steps=steps)
        assert state["steps"] == steps

    def test_chat_state_results_list(self):
        """Test results as list of strings."""
        results = ["Result 1", "Result 2"]
        state = ChatState(messages=[], results=results)
        assert state["results"] == results

    def test_chat_state_current_step_int(self):
        """Test current_step as integer."""
        state = ChatState(messages=[], current_step=5)
        assert state["current_step"] == 5

    def test_chat_state_pending_tasks_list(self):
        """Test pending_tasks as list."""
        tasks = ["task1", "task2", "task3"]
        state = ChatState(messages=[], pending_tasks=tasks)
        assert state["pending_tasks"] == tasks
