"""
Knowledge Assistant agent for answering questions from the knowledge base.

This agent is responsible for searching and retrieving information
from the knowledge base to answer user questions.
"""

from __future__ import annotations
from typing import Dict, Any, List, cast
from langchain_core.messages import AIMessage
from ..graph.state import GraphState
from ..graph.planning import get_next_task
from ..utils.mocks.data import search_knowledge
from ..graph.planning import mark_task_completed
from ..memory import get_ltm_service
from ..llm import should_use_mocks


class KnowledgeAssistant:
    """Knowledge Assistant for answering questions from the knowledge base."""

    def __init__(self):
        """Initialize the Knowledge Assistant."""
        pass

    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information."""
        return search_knowledge(query)

    async def generate_answer(self, query: str) -> str:
        """Generates an answer based on the retrieved knowledge."""
        answer_parts = []

        # First, search Long Term Memory for relevant historical Q&As (only in production mode)
        ltm_results = []
        if not should_use_mocks():
            try:
                ltm_service = get_ltm_service()
                ltm_results = await ltm_service.search_knowledge(query, limit=2)
            except Exception:
                # Don't fail if LTM search fails
                pass

        # If we found relevant results in LTM, use them to enhance the answer
        if ltm_results:
            answer_parts.append("📚 Based on previous interactions:")
            for i, result in enumerate(ltm_results[:2], 1):
                memory = result.memory
                # Extract key insights from historical responses
                response_preview = (
                    memory.system_response[:200] + "..."
                    if len(memory.system_response) > 200
                    else memory.system_response
                )
                similarity = round(result.similarity_score, 2)
                answer_parts.append(
                    f"{i}. (Similarity: {similarity}) {response_preview}"
                )

            answer_parts.append("")  # Add spacing

        # Then search the static knowledge base
        search_results = self.search_knowledge(query)

        if search_results:
            if ltm_results:
                answer_parts.append("📖 Additional information from knowledge base:")

            for i, search_item in enumerate(search_results, 1):
                search_result = cast(Dict[str, Any], search_item)
                answer_parts.append(
                    f"{search_result['term'].title()}: {search_result['content']}"
                )

                if search_result.get("related_topics"):
                    related = ", ".join(search_result["related_topics"])
                    answer_parts.append(f"Related topics: {related}")

                if i < len(search_results):
                    answer_parts.append("")  # Add spacing between results

        elif not ltm_results:
            # No results from either source
            return "I don't have information about that topic in my knowledge base or previous interactions. Please try rephrasing your question or ask about API, jobs, authentication, templates, or debugging."

        return "\n".join(answer_parts)


async def knowledge_assistant_node(state: GraphState) -> GraphState:
    """Knowledge Assistant node that answers questions from the knowledge base."""
    if not state["messages"]:
        # No messages - set route to response_synthesizer
        new_state = state.copy()
        new_state["route"] = "response_synthesizer"
        return new_state

    todo_list = state.get("todo_list", [])
    if not todo_list:
        return state

    # Get the next task for knowledge assistant
    next_task = await get_next_task(todo_list)
    if not next_task or next_task["agent"] != "knowledge_assistant":
        return state

    # Get query from task parameters or last message
    query = next_task["parameters"].get("query", "")
    if not query:
        # Fallback to last message
        last_message = state["messages"][-1]
        content = last_message.content
        if isinstance(content, list):
            text = ""
            for item in content:
                if isinstance(item, str):
                    text = item
                    break
        else:
            text = content or ""
        query = text

    # Create knowledge assistant and get answer
    assistant = KnowledgeAssistant()
    answer = await assistant.generate_answer(query)

    new_state = state.copy()
    msgs = list(new_state["messages"])
    msgs.append(AIMessage(content=f"📚 Knowledge Assistant: {answer}"))
    msgs.append(AIMessage(content=f"📋 Task: {next_task['description']}"))

    # Mark task as completed
    todo_list = mark_task_completed(todo_list, next_task["id"], answer)
    new_state["todo_list"] = todo_list

    new_state["messages"] = msgs

    # Set next route - go to response_synthesizer to format final response
    new_state["route"] = "response_synthesizer"

    return new_state
