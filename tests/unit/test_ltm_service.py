"""
Tests for LTM service functionality.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.multi_agent.memory.ltm_service import LTMService, get_ltm_service
from src.multi_agent.memory.schema import MemoryEntry, MemoryType, SearchResult
from src.multi_agent.graph.state import GraphState


class TestLTMService:
    """Test cases for LTMService."""

    @pytest.fixture
    def ltm_service(self):
        """Create an LTM service for testing."""
        return LTMService()

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        mock = AsyncMock()
        mock.store_memory = AsyncMock(return_value="test-id-123")
        mock.search_memories = AsyncMock(return_value=[])
        mock.get_recent_memories = AsyncMock(return_value=[])
        mock.get_memory_count = AsyncMock(return_value=0)
        return mock

    def test_ltm_service_initialization(self, ltm_service):
        """Test LTM service initialization."""
        assert ltm_service.vector_store is not None

    @pytest.mark.asyncio
    async def test_store_qa_session(self, ltm_service, mock_vector_store):
        """Test storing a Q&A session."""
        ltm_service.vector_store = mock_vector_store

        memory_id = await ltm_service.store_qa_session(
            user_query="What is the API?",
            system_response="The API is a RESTful service.",
            context={"route": "knowledge_assistant"},
        )

        assert memory_id == "test-id-123"
        mock_vector_store.store_memory.assert_called_once()

        # Verify the memory entry passed to store_memory
        call_args = mock_vector_store.store_memory.call_args[0][0]
        assert call_args.type == MemoryType.QA_SESSION
        assert call_args.user_query == "What is the API?"
        assert call_args.system_response == "The API is a RESTful service."
        assert call_args.context == {"route": "knowledge_assistant"}

    @pytest.mark.asyncio
    async def test_store_debug_analysis(self, ltm_service, mock_vector_store):
        """Test storing debug analysis."""
        ltm_service.vector_store = mock_vector_store

        analysis = {
            "error_code": "TEMPLATE_NOT_FOUND",
            "confidence_level": "High",
            "severity": "High",
        }

        memory_id = await ltm_service.store_debug_analysis(
            analysis=analysis,
            user_query="Debug job_003",
            response="Root cause analysis completed",
        )

        assert memory_id == "test-id-123"
        mock_vector_store.store_memory.assert_called_once()

        # Verify the memory entry
        call_args = mock_vector_store.store_memory.call_args[0][0]
        assert call_args.type == MemoryType.DEBUG_ANALYSIS
        assert call_args.error_code == "TEMPLATE_NOT_FOUND"
        assert call_args.confidence_level == "High"
        assert call_args.severity == "High"

    @pytest.mark.asyncio
    async def test_store_api_operation(self, ltm_service, mock_vector_store):
        """Test storing API operation."""
        ltm_service.vector_store = mock_vector_store

        memory_id = await ltm_service.store_api_operation(
            operation="run_job",
            user_query="run job data_processing",
            response="Job executed successfully",
            success=True,
        )

        assert memory_id == "test-id-123"
        mock_vector_store.store_memory.assert_called_once()

        # Verify the memory entry
        call_args = mock_vector_store.store_memory.call_args[0][0]
        assert call_args.type == MemoryType.API_OPERATION
        assert call_args.api_operation == "run_job"
        assert call_args.success is True

    @pytest.mark.asyncio
    async def test_search_similar_errors(self, ltm_service, mock_vector_store):
        """Test searching for similar errors."""
        # Mock search results
        mock_memory = MemoryEntry(
            type=MemoryType.DEBUG_ANALYSIS,
            user_query="Debug similar error",
            system_response="Similar analysis",
            error_code="TEMPLATE_NOT_FOUND",
        )
        mock_result = SearchResult(memory=mock_memory, similarity_score=0.9, rank=1)
        mock_vector_store.search_memories = AsyncMock(return_value=[mock_result])

        ltm_service.vector_store = mock_vector_store

        results = await ltm_service.search_similar_errors(
            error_code="TEMPLATE_NOT_FOUND", error_message="Template not found"
        )

        assert len(results) == 1
        assert results[0].memory.error_code == "TEMPLATE_NOT_FOUND"
        mock_vector_store.search_memories.assert_called_once_with(
            query="TEMPLATE_NOT_FOUND Template not found",
            memory_type=MemoryType.DEBUG_ANALYSIS,
            limit=3,
        )

    @pytest.mark.asyncio
    async def test_search_knowledge(self, ltm_service, mock_vector_store):
        """Test searching for knowledge."""
        # Mock search results for both QA and knowledge queries
        mock_vector_store.search_memories = AsyncMock(return_value=[])
        ltm_service.vector_store = mock_vector_store

        results = await ltm_service.search_knowledge("API documentation")

        assert isinstance(results, list)
        # Should call search_memories twice (for QA sessions and knowledge queries)
        assert mock_vector_store.search_memories.call_count == 2

    @pytest.mark.asyncio
    async def test_store_from_state_qa_session(self, ltm_service, mock_vector_store):
        """Test storing from graph state with Q&A session."""
        ltm_service.vector_store = mock_vector_store

        # Mock messages
        mock_message = MagicMock()
        mock_message.content = "What is the API?"

        state = GraphState(
            messages=[mock_message],
            final_response="The API is a RESTful service",
            route="knowledge_assistant",
        )

        stored_ids = await ltm_service.store_from_state(state)

        assert len(stored_ids) == 2  # Should store knowledge query + Q&A session
        assert all(sid == "test-id-123" for sid in stored_ids)

    @pytest.mark.asyncio
    async def test_store_from_state_with_debug_analysis(
        self, ltm_service, mock_vector_store
    ):
        """Test storing from graph state with debug analysis."""
        ltm_service.vector_store = mock_vector_store

        # Mock messages
        mock_message = MagicMock()
        mock_message.content = "Debug job_003"

        state = GraphState(
            messages=[mock_message],
            final_response="Debug analysis completed",
            root_cause_analysis={
                "error_code": "TEMPLATE_NOT_FOUND",
                "confidence_level": "High",
            },
        )

        stored_ids = await ltm_service.store_from_state(state)

        assert len(stored_ids) == 2  # Debug analysis + Q&A session
        mock_vector_store.store_memory.assert_called()

    @pytest.mark.asyncio
    async def test_get_stats(self, ltm_service, mock_vector_store):
        """Test getting LTM statistics."""
        mock_vector_store.get_memory_count = AsyncMock(return_value=42)
        mock_vector_store._initialized = True
        ltm_service.vector_store = mock_vector_store

        stats = await ltm_service.get_stats()

        assert stats["total_memories"] == 42
        assert stats["vector_store_initialized"] is True

    def test_get_ltm_service_singleton(self):
        """Test that get_ltm_service returns a singleton."""
        service1 = get_ltm_service()
        service2 = get_ltm_service()

        assert service1 is service2
