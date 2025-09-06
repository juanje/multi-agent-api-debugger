"""
Unit tests for tools.py
"""

from multi_agent.tools import (
    tool_calculator,
    tool_time,
    mock_retrieve,
    mock_rag_answer,
    detect_parallel_operations,
    detect_sequential_operations,
)


class TestToolCalculator:
    """Tests for the tool_calculator function."""

    def test_addition(self):
        """Test basic addition."""
        result = tool_calculator("5 + 3")
        assert "5+3 = 8" in result

    def test_subtraction(self):
        """Test basic subtraction."""
        result = tool_calculator("10 - 4")
        assert "10-4 = 6" in result

    def test_multiplication(self):
        """Test basic multiplication."""
        result = tool_calculator("6 * 7")
        assert "6*7 = 42" in result

    def test_multiplication_with_x(self):
        """Test multiplication with 'x'."""
        result = tool_calculator("3 x 4")
        assert "3x4 = 12" in result

    def test_division(self):
        """Test basic division."""
        result = tool_calculator("15 / 3")
        assert "15/3 = 5.0" in result

    def test_division_by_zero(self):
        """Test division by zero."""
        result = tool_calculator("5 / 0")
        assert "5/0 = ∞" in result

    def test_invalid_operation(self):
        """Test invalid operation."""
        result = tool_calculator("hello world")
        assert "I didn't find any simple operation" in result


class TestToolTime:
    """Tests for the tool_time function."""

    def test_time_format(self):
        """Test time format."""
        result = tool_time("time")
        assert "Current time (mock):" in result
        assert "2024" in result or "2025" in result  # Current year


class TestMockRetrieve:
    """Tests for the mock_retrieve function."""

    def test_retrieve_docs(self):
        """Test document retrieval."""
        docs = mock_retrieve("test query")
        assert len(docs) == 2
        assert all("title" in doc for doc in docs)
        assert all("snippet" in doc for doc in docs)
        assert "test query" in docs[0]["snippet"]


class TestMockRagAnswer:
    """Tests for the mock_rag_answer function."""

    def test_rag_answer(self):
        """Test RAG answer."""
        docs = [{"snippet": "Test content 1"}, {"snippet": "Test content 2"}]
        result = mock_rag_answer("test query", docs)
        assert "test query" in result
        assert "Test content 1" in result
        assert "Test content 2" in result
        assert "(RAG-mock)" in result


class TestDetectParallelOperations:
    """Tests for detect_parallel_operations."""

    def test_parallel_calculations(self):
        """Test parallel calculations detection."""
        result = detect_parallel_operations("What is 2+3 and 8/2?")
        assert result == ["2+3", "8/2"]

    def test_parallel_calc_and_time(self):
        """Test calculation and time detection."""
        result = detect_parallel_operations("2+3 and time")
        assert result == ["2+3", "time"]

    def test_parallel_time_and_calc(self):
        """Test time and calculation detection."""
        result = detect_parallel_operations("time and 2+3")
        assert result == ["time", "2+3"]

    def test_parallel_time_and_rag(self):
        """Test time and RAG detection."""
        result = detect_parallel_operations("time and manual")
        assert result == ["time", "rag"]

    def test_no_parallel_operations(self):
        """Test without parallel operations."""
        result = detect_parallel_operations("hello world")
        assert result is None


class TestDetectSequentialOperations:
    """Tests for detect_sequential_operations."""

    def test_sequential_sum(self):
        """Test sequential sum detection."""
        result = detect_sequential_operations("sum of 3x8 and 129/3")
        assert result is not None
        assert len(result) == 3  # 2 calculations + 1 sum
        assert result[0]["type"] == "calc"
        assert result[0]["operation"] == "3x8"
        assert result[1]["type"] == "calc"
        assert result[1]["operation"] == "129/3"
        assert result[2]["type"] == "sum"

    def test_sequential_time_and_rag(self):
        """Test sequential time and RAG detection."""
        result = detect_sequential_operations(
            "search for information about tasks for this time"
        )
        assert result is not None
        assert len(result) == 2
        assert result[0]["type"] == "time"
        assert result[1]["type"] == "rag"
        assert result[1]["query"] == "tasks"

    def test_sequential_calc_and_rag(self):
        """Test sequential calculation and RAG detection."""
        result = detect_sequential_operations("calculate 5*3 and then search for info")
        assert result is not None
        assert len(result) == 2
        assert result[0]["type"] == "calc"
        assert result[0]["operation"] == "5*3"
        assert result[1]["type"] == "rag"
        assert result[1]["query"] == "info"

    def test_no_sequential_operations(self):
        """Test without sequential operations."""
        result = detect_sequential_operations("hello world")
        assert result is None
