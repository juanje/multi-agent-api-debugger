"""
Unit tests for Response Synthesizer.
"""

from langchain_core.messages import HumanMessage
from multi_agent.response_synthesizer import (
    response_synthesizer_node,
    ResponseSynthesizer,
)


class TestResponseSynthesizer:
    """Tests for ResponseSynthesizer class."""

    def test_format_api_success_response(self):
        """Test formatting API success response."""
        synthesizer = ResponseSynthesizer()
        state = {
            "results": {
                "list_public_jobs": {
                    "jobs": [
                        {"id": "job_001", "name": "Test Job", "status": "completed"}
                    ]
                }
            }
        }

        response = synthesizer.synthesize_response(state)

        assert "API Operation Completed" in response
        assert "list_public_jobs" in response

    def test_format_api_error_response(self):
        """Test formatting API error response."""
        synthesizer = ResponseSynthesizer()
        state = {
            "error_info": {
                "error_code": "TEMPLATE_NOT_FOUND",
                "error_message": "Template not found",
                "timestamp": "2024-01-15T10:00:00Z",
                "suggestions": ["Check template name"],
            },
            "results": {"run_job": {"error": "Template not found"}},
        }

        response = synthesizer.synthesize_response(state)

        assert "API Operation Failed" in response
        assert "TEMPLATE_NOT_FOUND" in response

    def test_format_debugging_response(self):
        """Test formatting debugging response."""
        synthesizer = ResponseSynthesizer()
        state = {
            "error_info": {"error_code": "TEMPLATE_NOT_FOUND"},
            "root_cause_analysis": {
                "error_code": "TEMPLATE_NOT_FOUND",
                "root_cause_hypothesis": "Template missing",
                "confidence_level": "High",
                "recommended_actions": ["Check template"],
                "related_components": ["Template Engine"],
            },
        }

        response = synthesizer.synthesize_response(state)

        assert "Debugging Analysis Complete" in response
        assert "Template missing" in response

    def test_create_knowledge_summary(self):
        """Test creating knowledge summary."""
        synthesizer = ResponseSynthesizer()
        state = {
            "goal": "test goal",
            "results": {"list_public_jobs": {"jobs": []}},
            "error_info": None,
            "root_cause_analysis": None,
        }

        summary = synthesizer.create_knowledge_summary(state)

        assert len(summary) > 0
        assert summary[0]["type"] == "api_operations"
        assert "list_public_jobs" in summary[0]["operations_performed"]
        assert summary[0]["success"] is True


class TestResponseSynthesizerNode:
    """Tests for response_synthesizer_node function."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = response_synthesizer_node(state)
        assert result == state

    def test_synthesize_with_results(self):
        """Test synthesizing with results."""
        state = {
            "messages": [HumanMessage(content="test")],
            "results": {"list_public_jobs": {"jobs": []}},
        }
        result = response_synthesizer_node(state)

        assert "final_response" in result
        assert result["route"] == "done"
        assert "knowledge_summary" in result
