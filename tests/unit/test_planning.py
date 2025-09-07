"""
Unit tests for the planning module.

Note: These tests use the mocking system to test the planning functions
without requiring real LLM calls, making tests fast and deterministic.
"""

import os
import pytest
from langchain_core.messages import HumanMessage
from multi_agent.graph.planning import (
    create_comprehensive_todo_list,
    get_next_task,
    mark_task_completed,
    mark_task_failed,
    get_pending_tasks,
    get_completed_tasks,
    get_failed_tasks,
    is_workflow_complete,
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


class TestCreateComprehensiveTodoList:
    """Test cases for create_comprehensive_todo_list function."""

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

        todo_list = await create_comprehensive_todo_list(state)
        assert len(todo_list) > 0
        assert todo_list[0]["agent"] == "api_operator"
        assert todo_list[0]["status"] == "pending"
        assert "id" in todo_list[0]
        assert "description" in todo_list[0]
        assert "parameters" in todo_list[0]

    @pytest.mark.asyncio
    async def test_create_debug_todo_list(self):
        """Test creating todo list for debug operations."""
        state = GraphState(
            messages=[HumanMessage(content="debug job_001 error")],
            todo_list=[],
            results={},
            final_response=None,
            error_info=None,
            root_cause_analysis=None,
        )

        todo_list = await create_comprehensive_todo_list(state)
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

        todo_list = await create_comprehensive_todo_list(state)
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

        todo_list = await create_comprehensive_todo_list(state)
        assert len(todo_list) == 0


class TestGetNextTask:
    """Test cases for get_next_task function."""

    @pytest.mark.asyncio
    async def test_get_next_pending_task(self):
        """Test getting next pending task."""
        todo_list = [
            {"id": "task_1", "status": "completed"},
            {"id": "task_2", "status": "pending"},
            {"id": "task_3", "status": "pending"},
        ]

        next_task = await get_next_task(todo_list)
        assert next_task is not None
        assert next_task["id"] == "task_2"
        assert next_task["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_next_task_empty_list(self):
        """Test getting next task from empty list."""
        todo_list = []

        next_task = await get_next_task(todo_list)
        assert next_task is None

    @pytest.mark.asyncio
    async def test_get_next_task_no_pending(self):
        """Test getting next task when no pending tasks."""
        todo_list = [
            {"id": "task_1", "status": "completed"},
            {"id": "task_2", "status": "failed"},
        ]

        next_task = await get_next_task(todo_list)
        assert next_task is None


class TestTaskManagement:
    """Test cases for task management functions."""

    def test_mark_task_completed(self):
        """Test marking task as completed."""
        todo_list = [
            {"id": "task_1", "status": "pending"},
            {"id": "task_2", "status": "pending"},
        ]

        updated_list = mark_task_completed(todo_list, "task_1")
        assert updated_list[0]["status"] == "completed"
        assert updated_list[1]["status"] == "pending"

    def test_mark_task_failed(self):
        """Test marking task as failed."""
        todo_list = [
            {"id": "task_1", "status": "pending"},
            {"id": "task_2", "status": "pending"},
        ]

        updated_list = mark_task_failed(todo_list, "task_1", "Test error")
        assert updated_list[0]["status"] == "failed"
        assert updated_list[0]["error"] == "Test error"
        assert updated_list[1]["status"] == "pending"

    def test_get_pending_tasks(self):
        """Test getting pending tasks."""
        todo_list = [
            {"id": "task_1", "status": "pending"},
            {"id": "task_2", "status": "completed"},
            {"id": "task_3", "status": "pending"},
        ]

        pending_tasks = get_pending_tasks(todo_list)
        assert len(pending_tasks) == 2
        assert pending_tasks[0]["id"] == "task_1"
        assert pending_tasks[1]["id"] == "task_3"

    def test_get_completed_tasks(self):
        """Test getting completed tasks."""
        todo_list = [
            {"id": "task_1", "status": "pending"},
            {"id": "task_2", "status": "completed"},
            {"id": "task_3", "status": "completed"},
        ]

        completed_tasks = get_completed_tasks(todo_list)
        assert len(completed_tasks) == 2
        assert completed_tasks[0]["id"] == "task_2"
        assert completed_tasks[1]["id"] == "task_3"

    def test_get_failed_tasks(self):
        """Test getting failed tasks."""
        todo_list = [
            {"id": "task_1", "status": "pending"},
            {"id": "task_2", "status": "failed"},
            {"id": "task_3", "status": "failed"},
        ]

        failed_tasks = get_failed_tasks(todo_list)
        assert len(failed_tasks) == 2
        assert failed_tasks[0]["id"] == "task_2"
        assert failed_tasks[1]["id"] == "task_3"

    def test_is_workflow_complete(self):
        """Test checking if workflow is complete."""
        # All tasks completed
        todo_list = [
            {"id": "task_1", "status": "completed"},
            {"id": "task_2", "status": "completed"},
        ]
        assert is_workflow_complete(todo_list) is True

        # Some tasks pending
        todo_list = [
            {"id": "task_1", "status": "completed"},
            {"id": "task_2", "status": "pending"},
        ]
        assert is_workflow_complete(todo_list) is False

        # Some tasks failed
        todo_list = [
            {"id": "task_1", "status": "completed"},
            {"id": "task_2", "status": "failed"},
        ]
        assert is_workflow_complete(todo_list) is True

        # Empty list
        todo_list = []
        assert is_workflow_complete(todo_list) is True
