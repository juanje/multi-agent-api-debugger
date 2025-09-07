"""
Unit tests for Response Synthesizer.
"""

import os
import pytest
from langchain_core.messages import HumanMessage
from multi_agent.agents.response_synthesizer import (
    response_synthesizer_node,
    ResponseSynthesizer,
)


@pytest.fixture(autouse=True)
def enable_mocking():
    """Enable LLM mocking for all tests in this module."""
    os.environ["USE_LLM_MOCKS"] = "true"
    yield
    # Clean up after test
    if "USE_LLM_MOCKS" in os.environ:
        del os.environ["USE_LLM_MOCKS"]


class TestResponseSynthesizer:
    """Tests for ResponseSynthesizer class."""

    async def test_format_api_success_response(self):
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

        response = await synthesizer.synthesize_response(state)

        # Check that response contains job-related content
        assert any(
            word in response.lower() for word in ["job", "completed", "test job"]
        )
        assert len(response) > 50  # Ensure substantive response

    async def test_format_api_error_response(self):
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

        response = await synthesizer.synthesize_response(state)

        # Check that response contains error-related content
        assert any(
            word in response.lower()
            for word in ["error", "template", "issue", "problem"]
        )
        assert "TEMPLATE_NOT_FOUND" in response
        assert len(response) > 50  # Ensure substantive response

    async def test_format_debugging_response(self):
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

        response = await synthesizer.synthesize_response(state)

        # Check that response contains debugging-related content
        assert any(
            word in response.lower()
            for word in ["template", "missing", "issue", "error"]
        )
        assert "TEMPLATE_NOT_FOUND" in response
        assert len(response) > 50  # Ensure substantive response

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

    async def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = await response_synthesizer_node(state)
        # With our fixes, empty messages should set route to done
        assert result["route"] == "done"

    async def test_synthesize_with_results(self):
        """Test synthesizing with results."""
        state = {
            "messages": [HumanMessage(content="test")],
            "results": {"list_public_jobs": {"jobs": []}},
        }
        result = await response_synthesizer_node(state)

        assert "final_response" in result
        assert result["route"] == "done"
        assert "knowledge_summary" in result
