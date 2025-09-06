"""
Unit tests for Knowledge Assistant.
"""

from langchain_core.messages import HumanMessage
from multi_agent.knowledge_assistant import knowledge_assistant_node, KnowledgeAssistant


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

    def test_generate_answer(self):
        """Test generating answer from search results."""
        assistant = KnowledgeAssistant()
        answer = assistant.generate_answer("API")

        assert "API" in answer
        assert "endpoints" in answer

    def test_generate_answer_no_results(self):
        """Test generating answer with no search results."""
        assistant = KnowledgeAssistant()
        answer = assistant.generate_answer("completely unknown topic")

        assert "don't have information" in answer.lower()


class TestKnowledgeAssistantNode:
    """Tests for knowledge_assistant_node function."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = {"messages": []}
        result = knowledge_assistant_node(state)
        assert result == state

    def test_api_question(self):
        """Test with API question."""
        from multi_agent.mocks.planning import create_task

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
        result = knowledge_assistant_node(state)

        assert len(result["messages"]) > 1
        # Should have the original message plus assistant response
        assert any("API" in str(msg.content) for msg in result["messages"][1:])

    def test_jobs_question(self):
        """Test with jobs question."""
        from multi_agent.mocks.planning import create_task

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
        result = knowledge_assistant_node(state)

        assert len(result["messages"]) > 1
        assert any("jobs" in str(msg.content).lower() for msg in result["messages"][1:])
