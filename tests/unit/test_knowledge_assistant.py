"""
Unit tests for Knowledge Assistant.
"""

import os
import pytest
from langchain_core.messages import HumanMessage
from multi_agent.agents.knowledge_assistant import (
    knowledge_assistant_node,
    KnowledgeAssistant,
)


@pytest.fixture(autouse=True)
def enable_mocking():
    """Enable LLM mocking for all tests in this module."""
    os.environ["USE_LLM_MOCKS"] = "true"
    yield
    # Clean up after test
    if "USE_LLM_MOCKS" in os.environ:
        del os.environ["USE_LLM_MOCKS"]


class TestKnowledgeAssistant:
    """Tests for KnowledgeAssistant class."""

    def test_search_knowledge_api(self):
        """Test searching for API knowledge."""
        assistant = KnowledgeAssistant()
        results = assistant.search_knowledge("API")

        assert len(results) > 0
        assert any("api" in result["term"].lower() for result in results)

    def test_search_knowledge_jobs(self):
        """Test searching for jobs knowledge."""
        assistant = KnowledgeAssistant()
        results = assistant.search_knowledge("jobs")

        assert len(results) > 0
        assert any("jobs" in result["term"].lower() for result in results)

    def test_search_knowledge_no_results(self):
        """Test searching with no results."""
        assistant = KnowledgeAssistant()
        results = assistant.search_knowledge("completely unknown topic")

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_generate_answer(self):
        """Test generating answer from search results."""
        assistant = KnowledgeAssistant()
        answer = await assistant.generate_answer("API")

        assert "API" in answer
        assert "endpoints" in answer

    @pytest.mark.asyncio
    async def test_generate_answer_no_results(self):
        """Test generating answer with no search results."""
        assistant = KnowledgeAssistant()
        answer = await assistant.generate_answer("completely unknown topic")

        assert "don't have information" in answer.lower()


class TestKnowledgeAssistantNode:
    """Tests for knowledge_assistant_node function."""

    async def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = await knowledge_assistant_node(state)
        # With our fixes, empty messages should set route to response_synthesizer
        assert result["route"] == "response_synthesizer"

    async def test_api_question(self):
        """Test with API question."""
        from multi_agent.utils.mocks.planning import create_task

        state = {
            "messages": [HumanMessage(content="what is the API?")],
            "todo_list": [
                create_task(
                    description="Answer knowledge question",
                    agent="knowledge_assistant",
                    parameters={"query": "what is the API?"},
                )
            ],
        }
        result = await knowledge_assistant_node(state)

        assert len(result["messages"]) > 1
        # Should have the original message plus assistant response
        assert any("API" in str(msg.content) for msg in result["messages"][1:])

    async def test_jobs_question(self):
        """Test with jobs question."""
        from multi_agent.utils.mocks.planning import create_task

        state = {
            "messages": [HumanMessage(content="tell me about jobs")],
            "todo_list": [
                create_task(
                    description="Answer knowledge question",
                    agent="knowledge_assistant",
                    parameters={"query": "tell me about jobs"},
                )
            ],
        }
        result = await knowledge_assistant_node(state)

        assert len(result["messages"]) > 1
        assert any("jobs" in str(msg.content).lower() for msg in result["messages"][1:])
