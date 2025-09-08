"""
Tests for memory schema and models.
"""

from src.multi_agent.memory.schema import MemoryEntry, MemoryType, SearchResult


class TestMemoryEntry:
    """Test cases for MemoryEntry model."""

    def test_memory_entry_creation(self):
        """Test creating a basic memory entry."""
        memory = MemoryEntry(
            type=MemoryType.QA_SESSION,
            user_query="What is the API?",
            system_response="The API is a RESTful service for job management.",
        )

        assert memory.type == MemoryType.QA_SESSION
        assert memory.user_query == "What is the API?"
        assert (
            memory.system_response == "The API is a RESTful service for job management."
        )
        assert memory.context == {}
        assert memory.metadata == {}
        assert memory.id is None  # Should be None initially

    def test_memory_entry_with_debug_fields(self):
        """Test creating a debug analysis memory entry."""
        memory = MemoryEntry(
            type=MemoryType.DEBUG_ANALYSIS,
            user_query="Debug job_003",
            system_response="Analysis completed",
            error_code="TEMPLATE_NOT_FOUND",
            confidence_level="High",
            severity="High",
        )

        assert memory.type == MemoryType.DEBUG_ANALYSIS
        assert memory.error_code == "TEMPLATE_NOT_FOUND"
        assert memory.confidence_level == "High"
        assert memory.severity == "High"

    def test_memory_entry_with_api_fields(self):
        """Test creating an API operation memory entry."""
        memory = MemoryEntry(
            type=MemoryType.API_OPERATION,
            user_query="run job data_processing",
            system_response="Job executed successfully",
            api_operation="run_job",
            success=True,
        )

        assert memory.type == MemoryType.API_OPERATION
        assert memory.api_operation == "run_job"
        assert memory.success is True

    def test_get_searchable_text(self):
        """Test getting searchable text from memory entry."""
        memory = MemoryEntry(
            type=MemoryType.DEBUG_ANALYSIS,
            user_query="Debug error",
            system_response="Error analysis",
            error_code="TEST_ERROR",
            api_operation="test_op",
            related_topics=["debugging", "errors"],
        )

        searchable_text = memory.get_searchable_text()
        expected_parts = [
            "Debug error",
            "Error analysis",
            "Error: TEST_ERROR",
            "API Operation: test_op",
            "Topics: debugging, errors",
        ]

        for part in expected_parts:
            assert part in searchable_text

    def test_get_metadata_dict(self):
        """Test getting metadata dictionary."""
        memory = MemoryEntry(
            type=MemoryType.QA_SESSION,
            user_query="Test query",
            system_response="Test response",
            metadata={"custom": "value"},
        )

        metadata_dict = memory.get_metadata_dict()
        assert metadata_dict["type"] == "qa_session"
        assert "timestamp" in metadata_dict
        assert metadata_dict["custom"] == "value"

    def test_memory_type_enum(self):
        """Test MemoryType enum values."""
        assert MemoryType.QA_SESSION == "qa_session"
        assert MemoryType.DEBUG_ANALYSIS == "debug_analysis"
        assert MemoryType.API_OPERATION == "api_operation"
        assert MemoryType.KNOWLEDGE_QUERY == "knowledge_query"


class TestSearchResult:
    """Test cases for SearchResult model."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        memory = MemoryEntry(
            type=MemoryType.QA_SESSION, user_query="Test", system_response="Response"
        )

        result = SearchResult(memory=memory, similarity_score=0.85, rank=1)

        assert result.memory == memory
        assert result.similarity_score == 0.85
        assert result.rank == 1
