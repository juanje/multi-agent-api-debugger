"""
Unit tests for executors.py
"""

import pytest
from multi_agent.executors import (
    execute_operation_async,
    parallel_executor_node,
    sequential_executor_node,
    finish_node,
)


class TestExecuteOperationAsync:
    """Tests for execute_operation_async."""

    @pytest.mark.asyncio
    async def test_calculation_operation(self):
        """Test calculation operation execution."""
        result = await execute_operation_async("5 + 3")
        assert "5+3 = 8" in result

    @pytest.mark.asyncio
    async def test_time_operation(self):
        """Test time operation execution."""
        result = await execute_operation_async("time")
        assert "Current time (mock):" in result

    @pytest.mark.asyncio
    async def test_rag_operation(self):
        """Test RAG operation execution."""
        result = await execute_operation_async("rag")
        assert "(RAG-mock)" in result
        assert "manual" in result

    @pytest.mark.asyncio
    async def test_document_operation(self):
        """Test document operation execution."""
        result = await execute_operation_async("search the manual")
        assert "(RAG-mock)" in result
        assert "search the manual" in result

    @pytest.mark.asyncio
    async def test_unknown_operation(self):
        """Test unknown operation execution."""
        result = await execute_operation_async("unknown operation")
        assert "Unrecognized operation" in result


class TestParallelExecutorNode:
    """Tests for parallel_executor_node."""

    def test_empty_operations(self):
        """Test with empty operations."""
        state = {"operations": []}
        result = parallel_executor_node(state)
        assert result == state

    def test_single_operation(self):
        """Test with single operation."""
        state = {
            "operations": ["5 + 3"],
            "messages": [],
        }
        result = parallel_executor_node(state)
        assert len(result["messages"]) >= 2  # Intermediate + final
        assert "Executing operations in parallel" in result["messages"][-2].content

    def test_multiple_operations(self):
        """Test with multiple operations."""
        state = {
            "operations": ["2+3", "8/2"],
            "messages": [],
        }
        result = parallel_executor_node(state)
        assert len(result["messages"]) >= 2
        assert "Executing operations in parallel" in result["messages"][-2].content
        assert "5 and 4.0" in result["messages"][-1].content

    def test_mixed_operations(self):
        """Test with mixed operations."""
        state = {
            "operations": ["2+3", "time"],
            "messages": [],
        }
        result = parallel_executor_node(state)
        assert len(result["messages"]) >= 2
        assert "Executing operations in parallel" in result["messages"][-2].content


class TestSequentialExecutorNode:
    """Tests for sequential_executor_node."""

    def test_no_steps(self):
        """Test without steps."""
        state = {"steps": [], "current_step": 0, "results": []}
        result = sequential_executor_node(state)
        assert result == state

    def test_current_step_beyond_steps(self):
        """Test when current_step is beyond steps."""
        state = {
            "steps": [{"type": "calc", "operation": "2+3", "step": 1}],
            "current_step": 1,
            "results": [],
        }
        result = sequential_executor_node(state)
        assert result == state

    def test_calc_step(self):
        """Test calculation step execution."""
        state = {
            "steps": [{"type": "calc", "operation": "3x8", "step": 1}],
            "current_step": 0,
            "results": [],
            "messages": [],
        }
        result = sequential_executor_node(state)
        assert result["current_step"] == 1
        assert len(result["results"]) == 1
        assert "3x8 = 24" in result["results"][0]
        assert "Executing operations sequentially" in result["messages"][-1].content

    def test_time_step(self):
        """Test time step execution."""
        state = {
            "steps": [{"type": "time", "step": 1}],
            "current_step": 0,
            "results": [],
            "messages": [],
        }
        result = sequential_executor_node(state)
        assert result["current_step"] == 1
        assert len(result["results"]) == 1
        assert "Current time (mock):" in result["results"][0]

    def test_rag_step(self):
        """Test RAG step execution."""
        state = {
            "steps": [{"type": "rag", "query": "test", "step": 1}],
            "current_step": 0,
            "results": [],
            "messages": [],
        }
        result = sequential_executor_node(state)
        assert result["current_step"] == 1
        assert len(result["results"]) == 1
        assert "(RAG-mock)" in result["results"][0]

    def test_sum_step(self):
        """Test sum step execution."""
        state = {
            "steps": [{"type": "sum", "step": 1}],
            "current_step": 0,
            "results": [
                "Result (mock calc): 3x8 = 24",
                "Result (mock calc): 129/3 = 43.0",
            ],
            "messages": [],
        }
        result = sequential_executor_node(state)
        assert result["current_step"] == 1
        assert len(result["results"]) == 3
        assert "Total sum: 24 + 43 = 67" in result["results"][-1]

    def test_sum_step_no_calc_results(self):
        """Test sum step without calculation results."""
        state = {
            "steps": [{"type": "sum", "step": 1}],
            "current_step": 0,
            "results": ["Other result"],
            "messages": [],
        }
        result = sequential_executor_node(state)
        assert result["current_step"] == 1
        assert len(result["results"]) == 2
        assert "No calculation results to sum" in result["results"][-1]


class TestFinishNode:
    """Tests for finish_node."""

    def test_finish_node(self):
        """Test finish node."""
        state = {"test": "value"}
        result = finish_node(state)
        assert result == state
