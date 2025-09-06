"""
Unit tests for API Operator.
"""

from langchain_core.messages import HumanMessage
from multi_agent.api_operator import api_operator_node, APIOperator


class TestAPIOperator:
    """Tests for APIOperator class."""

    def test_list_public_jobs(self):
        """Test listing public jobs."""
        operator = APIOperator()
        result = operator.execute_tool("list_public_jobs", {})

        assert "jobs" in result
        assert isinstance(result["jobs"], list)
        assert len(result["jobs"]) > 0

    def test_run_job(self):
        """Test running a job."""
        operator = APIOperator()
        result = operator.execute_tool("run_job", {"job_name": "test_job"})

        assert "job_id" in result
        assert "status" in result
        assert result["job_name"] == "test_job"

    def test_get_job_results(self):
        """Test getting job results."""
        operator = APIOperator()
        result = operator.execute_tool("get_job_results", {"job_id": "job_001"})

        assert "job_id" in result
        assert "status" in result

    def test_get_job_results_error(self):
        """Test getting job results for failed job."""
        operator = APIOperator()
        result = operator.execute_tool("get_job_results", {"job_id": "job_003"})

        assert "job_id" in result
        assert result["status"] == "failed"
        assert "error" in result

    def test_check_system_status(self):
        """Test checking system status."""
        operator = APIOperator()
        result = operator.execute_tool("check_system_status", {})

        assert "status" in result
        assert "active_jobs" in result

    def test_unknown_tool(self):
        """Test unknown tool."""
        operator = APIOperator()
        result = operator.execute_tool("unknown_tool", {})

        assert "error" in result
        assert "Unknown tool" in result["error"]


class TestAPIOperatorNode:
    """Tests for api_operator_node function."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = api_operator_node(state)
        assert result == state

    def test_no_todo_list(self):
        """Test with no todo list."""
        state = {"messages": [HumanMessage(content="test")]}
        result = api_operator_node(state)

        # Should return state unchanged when no todo list
        assert result == state

    def test_execute_tool(self):
        """Test executing a tool from todo list."""
        from multi_agent.mocks.planning import create_task

        state = {
            "messages": [HumanMessage(content="list jobs")],
            "todo_list": [
                create_task(
                    description="List all available jobs",
                    agent="api_operator",
                    parameters={"operation": "list_public_jobs"},
                )
            ],
        }
        result = api_operator_node(state)

        assert "results" in result
        assert "list_public_jobs" in result["results"]
        # Check that the task was marked as completed
        assert result["todo_list"][0]["status"] == "completed"
