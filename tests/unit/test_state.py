"""
Unit tests for state.py
"""

from multi_agent.state import GraphState


class TestGraphState:
    """Tests for the GraphState class."""

    def test_graph_state_creation(self):
        """Test GraphState creation with new architecture fields."""
        state = GraphState(
            messages=[],
            goal="test goal",
            todo_list=["list_public_jobs"],
            results={"list_public_jobs": {"jobs": []}},
            error_info=None,
            root_cause_analysis=None,
            final_response=None,
            route="api_operator",
            next_agent="api_operator",
            knowledge_summary=[],
        )

        assert state["messages"] == []
        assert state["goal"] == "test goal"
        assert state["todo_list"] == ["list_public_jobs"]
        assert state["results"] == {"list_public_jobs": {"jobs": []}}
        assert state["error_info"] is None
        assert state["root_cause_analysis"] is None
        assert state["final_response"] is None
        assert state["route"] == "api_operator"
        assert state["next_agent"] == "api_operator"
        assert state["knowledge_summary"] == []

    def test_graph_state_optional_fields(self):
        """Test GraphState with optional fields."""
        state = GraphState(
            messages=[],
            goal=None,
            todo_list=None,
            results=None,
            error_info=None,
            root_cause_analysis=None,
            final_response=None,
            route=None,
            next_agent=None,
            knowledge_summary=None,
        )

        assert state["messages"] == []
        assert state["goal"] is None
        assert state["todo_list"] is None
        assert state["results"] is None
        assert state["error_info"] is None
        assert state["root_cause_analysis"] is None
        assert state["final_response"] is None
        assert state["route"] is None
        assert state["next_agent"] is None
        assert state["knowledge_summary"] is None

    def test_graph_state_inheritance(self):
        """Test that GraphState is a TypedDict."""
        assert hasattr(GraphState, "__annotations__")
        assert "messages" in GraphState.__annotations__
        assert "goal" in GraphState.__annotations__
        assert "route" in GraphState.__annotations__

    def test_graph_state_route_values(self):
        """Test valid values for route."""
        valid_routes = [
            "api_operator",
            "debugger",
            "knowledge_assistant",
            "response_synthesizer",
            "done",
        ]

        for route in valid_routes:
            state = GraphState(messages=[], route=route)
            assert state["route"] == route

    def test_graph_state_results_dict(self):
        """Test results as dictionary."""
        results = {"list_public_jobs": {"jobs": []}, "run_job": {"job_id": "job_001"}}
        state = GraphState(messages=[], results=results)
        assert state["results"] == results

    def test_graph_state_error_info(self):
        """Test error_info field."""
        error_info = {
            "error_code": "TEMPLATE_NOT_FOUND",
            "error_message": "Template not found",
        }
        state = GraphState(messages=[], error_info=error_info)
        assert state["error_info"] == error_info

    def test_graph_state_root_cause_analysis(self):
        """Test root_cause_analysis field."""
        analysis = {"error_code": "TEMPLATE_NOT_FOUND", "confidence_level": "High"}
        state = GraphState(messages=[], root_cause_analysis=analysis)
        assert state["root_cause_analysis"] == analysis

    def test_graph_state_todo_list(self):
        """Test todo_list field."""
        todo_list = ["list_public_jobs", "run_job:job_name=data_processing"]
        state = GraphState(messages=[], todo_list=todo_list)
        assert state["todo_list"] == todo_list

    def test_graph_state_knowledge_summary(self):
        """Test knowledge_summary field."""
        knowledge_summary = [
            {"topic": "API", "summary": "REST API for job management"},
            {"topic": "Jobs", "summary": "Background processing system"},
        ]
        state = GraphState(messages=[], knowledge_summary=knowledge_summary)
        assert state["knowledge_summary"] == knowledge_summary

    def test_graph_state_goal_field(self):
        """Test goal field."""
        goal = "Debug job_003 failure and provide solution"
        state = GraphState(messages=[], goal=goal)
        assert state["goal"] == goal

    def test_graph_state_next_agent_field(self):
        """Test next_agent field."""
        next_agent = "debugger"
        state = GraphState(messages=[], next_agent=next_agent)
        assert state["next_agent"] == next_agent

    def test_graph_state_final_response_field(self):
        """Test final_response field."""
        final_response = (
            "✅ Debugging Analysis Complete\n\nRoot Cause: Template missing"
        )
        state = GraphState(messages=[], final_response=final_response)
        assert state["final_response"] == final_response
