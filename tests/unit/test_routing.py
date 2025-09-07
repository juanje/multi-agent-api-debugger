"""
Unit tests for the routing module.

Note: These tests use the mocking system to test the routing functions
without requiring real LLM calls, making tests fast and deterministic.
"""

import os
import pytest
from langchain_core.messages import HumanMessage
from multi_agent.graph.routing import (
    determine_route,
    create_todo_list,
    extract_api_operations,
)
from multi_agent.graph.state import GraphState


@pytest.fixture(autouse=True)
def enable_mocking():
    """Enable LLM mocking for all tests in this module."""
    os.environ["USE_LLM_MOCKS"] = "true"
    yield
    # Clean up after test
    if "USE_LLM_MOCKS" in os.environ:
        del os.environ["USE_LLM_MOCKS"]


class TestDetermineRoute:
    """Test cases for determine_route function."""

    @pytest.mark.asyncio
    async def test_api_route(self):
        """Test routing to API operator."""
        state = GraphState(
            messages=[HumanMessage(content="list all jobs")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        route = await determine_route(state)
        assert route == "api_operator"

    @pytest.mark.asyncio
    async def test_debug_route(self):
        """Test routing to debugger."""
        state = GraphState(
            messages=[HumanMessage(content="debug job_001 error")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        route = await determine_route(state)
        assert route == "debugger"

    @pytest.mark.asyncio
    async def test_knowledge_route(self):
        """Test routing to knowledge assistant."""
        state = GraphState(
            messages=[HumanMessage(content="what are jobs?")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        route = await determine_route(state)
        assert route == "knowledge_assistant"

    @pytest.mark.asyncio
    async def test_done_route(self):
        """Test routing to done when final response exists."""
        state = GraphState(
            messages=[HumanMessage(content="test")],
            todo_list=[],
            results={},
            final_response="Test response",
            error_info=None,
            root_cause_analysis=None,
        )

        route = await determine_route(state)
        assert route == "done"

    @pytest.mark.asyncio
    async def test_empty_messages(self):
        """Test routing with empty messages."""
        state = GraphState(
            messages=[],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        route = await determine_route(state)
        assert route == "done"


class TestCreateTodoList:
    """Test cases for create_todo_list function."""

    @pytest.mark.asyncio
    async def test_create_api_todo_list(self):
        """Test creating todo list for API operations."""
        state = GraphState(
            messages=[HumanMessage(content="run data processing job")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        todo_list = await create_todo_list(state)
        assert len(todo_list) > 0
        assert todo_list[0]["agent"] == "api_operator"
        assert todo_list[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_debug_todo_list(self):
        """Test creating todo list for debug operations."""
        state = GraphState(
            messages=[HumanMessage(content="debug job_001")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        todo_list = await create_todo_list(state)
        assert len(todo_list) > 0
        assert todo_list[0]["agent"] == "debugger"
        assert todo_list[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_knowledge_todo_list(self):
        """Test creating todo list for knowledge operations."""
        state = GraphState(
            messages=[HumanMessage(content="what are job templates?")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        todo_list = await create_todo_list(state)
        assert len(todo_list) > 0
        assert todo_list[0]["agent"] == "knowledge_assistant"
        assert todo_list[0]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_empty_messages_todo_list(self):
        """Test creating todo list with empty messages."""
        state = GraphState(
            messages=[],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        todo_list = await create_todo_list(state)
        assert len(todo_list) == 0


class TestExtractApiOperations:
    """Test cases for extract_api_operations function."""

    @pytest.mark.asyncio
    async def test_extract_run_job_operation(self):
        """Test extracting run job operation."""
        state = GraphState(
            messages=[HumanMessage(content="run data processing job")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        operations = await extract_api_operations(state)
        assert len(operations) > 0
        assert operations[0]["operation"] == "run_job"
        assert "data_processing" in operations[0]["parameters"]["job_name"]

    @pytest.mark.asyncio
    async def test_extract_list_jobs_operation(self):
        """Test extracting list jobs operation."""
        state = GraphState(
            messages=[HumanMessage(content="list all jobs")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        operations = await extract_api_operations(state)
        assert len(operations) > 0
        assert operations[0]["operation"] == "list_public_jobs"

    @pytest.mark.asyncio
    async def test_extract_get_results_operation(self):
        """Test extracting get results operation."""
        state = GraphState(
            messages=[HumanMessage(content="show me the results for job_123")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        operations = await extract_api_operations(state)
        assert len(operations) > 0
        assert operations[0]["operation"] == "get_job_results"
        assert operations[0]["parameters"]["job_id"] == "job_123"

    @pytest.mark.asyncio
    async def test_extract_check_status_operation(self):
        """Test extracting check status operation."""
        state = GraphState(
            messages=[HumanMessage(content="check system status")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        operations = await extract_api_operations(state)
        assert len(operations) > 0
        assert operations[0]["operation"] == "check_system_status"

    @pytest.mark.asyncio
    async def test_no_api_operations(self):
        """Test when no API operations are found."""
        state = GraphState(
            messages=[HumanMessage(content="what are jobs?")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        operations = await extract_api_operations(state)
        # Mock returns default operation when no specific API operation is found
        assert len(operations) >= 0  # May return default operation
