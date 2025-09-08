"""
Integration tests for Long Term Memory (LTM) functionality.

These tests verify that LTM integrates properly with the agent workflow
and that memory storage and retrieval work end-to-end.
"""

import pytest
from unittest.mock import AsyncMock, patch
from src.multi_agent.agents.response_synthesizer import response_synthesizer_node
from src.multi_agent.agents.debugger import debugger_node
from src.multi_agent.agents.knowledge_assistant import knowledge_assistant_node
from src.multi_agent.graph.state import GraphState
from src.multi_agent.memory.ltm_service import get_ltm_service
from langchain_core.messages import HumanMessage


class TestLTMIntegration:
    """Integration tests for LTM with agent workflow."""

    @pytest.fixture
    def mock_ltm_service(self):
        """Create a mock LTM service."""
        mock = AsyncMock()
        mock.store_from_state = AsyncMock(return_value=["memory-id-123"])
        mock.search_similar_errors = AsyncMock(return_value=[])
        mock.search_knowledge = AsyncMock(return_value=[])
        return mock

    @pytest.mark.asyncio
    @patch(
        "src.multi_agent.agents.response_synthesizer.should_use_mocks",
        return_value=False,
    )
    @patch("src.multi_agent.agents.response_synthesizer.get_ltm_service")
    async def test_response_synthesizer_stores_in_ltm(
        self, mock_get_ltm_service, mock_should_use_mocks, mock_ltm_service
    ):
        """Test that response synthesizer stores interactions in LTM."""
        mock_get_ltm_service.return_value = mock_ltm_service

        # Create state with user interaction
        state = GraphState(
            messages=[HumanMessage(content="What is the API?")],
            final_response="",
            route="response_synthesizer",
        )

        # Run response synthesizer node
        result_state = await response_synthesizer_node(state)

        # Verify LTM service was called to store the interaction
        mock_ltm_service.store_from_state.assert_called_once()
        assert result_state["route"] == "done"
        assert "final_response" in result_state

    @pytest.mark.asyncio
    @patch("src.multi_agent.agents.debugger.should_use_mocks", return_value=False)
    @patch("src.multi_agent.agents.debugger.get_ltm_service")
    async def test_debugger_searches_ltm_for_similar_errors(
        self, mock_get_ltm_service, mock_should_use_mocks, mock_ltm_service
    ):
        """Test that debugger searches LTM for similar errors."""
        mock_get_ltm_service.return_value = mock_ltm_service

        # Create state with debug task
        todo_list = [
            {
                "id": "debug_001",
                "agent": "debugger",
                "status": "pending",
                "description": "Analyze error",
                "parameters": {"error_type": "template_not_found"},
            }
        ]

        state = GraphState(
            messages=[HumanMessage(content="Debug job_003")],
            todo_list=todo_list,
            route="debugger",
        )

        # Run debugger node
        result_state = await debugger_node(state)

        # Verify LTM service was called to search for similar errors
        mock_ltm_service.search_similar_errors.assert_called_once()
        assert "root_cause_analysis" in result_state
        assert result_state["route"] == "response_synthesizer"

    @pytest.mark.asyncio
    @patch(
        "src.multi_agent.agents.knowledge_assistant.should_use_mocks",
        return_value=False,
    )
    @patch("src.multi_agent.agents.knowledge_assistant.get_ltm_service")
    async def test_knowledge_assistant_searches_ltm(
        self, mock_get_ltm_service, mock_should_use_mocks, mock_ltm_service
    ):
        """Test that knowledge assistant searches LTM for historical Q&As."""
        mock_get_ltm_service.return_value = mock_ltm_service

        # Create state with knowledge task
        todo_list = [
            {
                "id": "knowledge_001",
                "agent": "knowledge_assistant",
                "status": "pending",
                "description": "Answer question",
                "parameters": {"query": "What is the API?"},
            }
        ]

        state = GraphState(
            messages=[HumanMessage(content="What is the API?")],
            todo_list=todo_list,
            route="knowledge_assistant",
        )

        # Run knowledge assistant node
        result_state = await knowledge_assistant_node(state)

        # Verify LTM service was called to search for knowledge
        mock_ltm_service.search_knowledge.assert_called_once()
        assert result_state["route"] == "response_synthesizer"

    @pytest.mark.asyncio
    @patch(
        "src.multi_agent.llm.should_use_mocks", return_value=True
    )  # Keep agents in mock mode
    async def test_end_to_end_workflow_with_ltm_storage(self, mock_should_use_mocks):
        """Test complete workflow stores interaction in LTM."""
        # Create initial state
        state = GraphState(
            messages=[HumanMessage(content="What is the API?")],
            route="response_synthesizer",
        )

        # Mock LTM service for this test
        with (
            patch(
                "src.multi_agent.agents.response_synthesizer.should_use_mocks",
                return_value=False,
            ),
            patch(
                "src.multi_agent.agents.response_synthesizer.get_ltm_service"
            ) as mock_get_ltm,
        ):
            mock_ltm_service = AsyncMock()
            mock_ltm_service.store_from_state = AsyncMock(
                return_value=["memory-id-123"]
            )
            mock_get_ltm.return_value = mock_ltm_service

            # Run response synthesizer (final step)
            result_state = await response_synthesizer_node(state)

            # Verify workflow completed and stored in LTM
            assert result_state["route"] == "done"
            assert "final_response" in result_state
            mock_ltm_service.store_from_state.assert_called_once()

    def test_ltm_service_singleton(self):
        """Test that LTM service maintains singleton pattern."""
        service1 = get_ltm_service()
        service2 = get_ltm_service()

        assert service1 is service2
        assert hasattr(service1, "vector_store")

    @pytest.mark.asyncio
    async def test_ltm_graceful_failure_handling(self):
        """Test that LTM failures don't break the workflow."""
        # Create state that would normally trigger LTM operations
        state = GraphState(
            messages=[HumanMessage(content="What is the API?")],
            route="response_synthesizer",
        )

        # Mock LTM service to raise exception
        with (
            patch(
                "src.multi_agent.agents.response_synthesizer.should_use_mocks",
                return_value=False,
            ),
            patch(
                "src.multi_agent.agents.response_synthesizer.get_ltm_service"
            ) as mock_get_ltm,
        ):
            mock_ltm_service = AsyncMock()
            mock_ltm_service.store_from_state = AsyncMock(
                side_effect=Exception("LTM failure")
            )
            mock_get_ltm.return_value = mock_ltm_service

            # Run response synthesizer - should not fail even if LTM fails
            result_state = await response_synthesizer_node(state)

            # Verify workflow still completed successfully
            assert result_state["route"] == "done"
            assert "final_response" in result_state
