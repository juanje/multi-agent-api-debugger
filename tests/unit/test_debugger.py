"""
Unit tests for Debugger.
"""

from langchain_core.messages import HumanMessage
from multi_agent.debugger import debugger_node, Debugger


class TestDebugger:
    """Tests for Debugger class."""

    def test_analyze_error_template_not_found(self):
        """Test analyzing TEMPLATE_NOT_FOUND error."""
        debugger = Debugger()
        error_info = {
            "error_code": "TEMPLATE_NOT_FOUND",
            "error_message": "Template not found",
            "timestamp": "2024-01-15T10:00:00Z",
            "suggestions": ["Check template name"],
        }

        analysis = debugger.analyze_error(error_info)

        assert analysis["error_code"] == "TEMPLATE_NOT_FOUND"
        assert "root_cause_hypothesis" in analysis
        assert "confidence_level" in analysis
        assert "recommended_actions" in analysis
        assert analysis["confidence_level"] == "High"

    def test_analyze_error_memory_limit(self):
        """Test analyzing MEMORY_LIMIT_EXCEEDED error."""
        debugger = Debugger()
        error_info = {
            "error_code": "MEMORY_LIMIT_EXCEEDED",
            "error_message": "Memory limit exceeded",
            "timestamp": "2024-01-15T10:00:00Z",
        }

        analysis = debugger.analyze_error(error_info)

        assert analysis["error_code"] == "MEMORY_LIMIT_EXCEEDED"
        assert analysis["confidence_level"] == "Medium"
        assert analysis["severity"] == "High"

    def test_analyze_unknown_error(self):
        """Test analyzing unknown error."""
        debugger = Debugger()
        error_info = {
            "error_code": "UNKNOWN_ERROR",
            "error_message": "Unknown error",
            "timestamp": "2024-01-15T10:00:00Z",
        }

        analysis = debugger.analyze_error(error_info)

        assert analysis["error_code"] == "UNKNOWN_ERROR"
        assert analysis["confidence_level"] == "Low"


class TestDebuggerNode:
    """Tests for debugger_node function."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = debugger_node(state)
        assert result == state

    def test_no_error_info(self):
        """Test with no error info."""
        state = {"messages": [HumanMessage(content="test")]}
        result = debugger_node(state)
        assert result == state

    def test_debug_job_003(self):
        """Test debugging job_003."""
        from multi_agent.mocks.planning import create_task

        state = {
            "messages": [HumanMessage(content="debug job_003")],
            "todo_list": [
                create_task(
                    description="Analyze error for job_003",
                    agent="debugger",
                    parameters={
                        "job_id": "job_003",
                        "error_type": "template_not_found",
                    },
                )
            ],
        }
        result = debugger_node(state)

        assert "error_info" in result
        assert "root_cause_analysis" in result
        assert result["error_info"]["error_code"] == "TEMPLATE_NOT_FOUND"

    def test_with_existing_error_info(self):
        """Test with existing error info."""
        from multi_agent.mocks.planning import create_task

        state = {
            "messages": [HumanMessage(content="test")],
            "error_info": {
                "error_code": "TIMEOUT_ERROR",
                "error_message": "Job timeout",
                "timestamp": "2024-01-15T10:00:00Z",
            },
            "todo_list": [
                create_task(
                    description="Analyze existing error",
                    agent="debugger",
                    parameters={"job_id": "job_001", "error_type": "timeout"},
                )
            ],
        }
        result = debugger_node(state)

        assert "root_cause_analysis" in result
        assert result["root_cause_analysis"]["error_code"] == "TIMEOUT_ERROR"
