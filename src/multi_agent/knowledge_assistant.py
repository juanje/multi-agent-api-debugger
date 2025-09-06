"""
Knowledge Assistant agent for answering questions from the knowledge base.

This agent is responsible for searching and retrieving information
from the knowledge base to answer user questions.
"""

from __future__ import annotations
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from multi_agent.state import GraphState
from multi_agent.mocks.data import search_knowledge
from multi_agent.mocks.planning import get_next_task, mark_task_completed


class KnowledgeAssistant:
    """Knowledge Assistant for answering questions from the knowledge base."""

    def __init__(self):
        """Initialize the Knowledge Assistant."""
        pass

    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information."""
        return search_knowledge(query)

    def generate_answer(self, query: str) -> str:
        """Generates an answer based on the retrieved knowledge."""
        search_results = self.search_knowledge(query)

        if not search_results:
            return "I don't have information about that topic in my knowledge base. Please try rephrasing your question or ask about API, jobs, authentication, templates, or debugging."

        # Build answer from search results
        answer_parts = []

        for i, result in enumerate(search_results, 1):
            answer_parts.append(f"{result['term'].title()}: {result['content']}")

            if result.get("related_topics"):
                related = ", ".join(result["related_topics"])
                answer_parts.append(f"Related topics: {related}")

            if i < len(search_results):
                answer_parts.append("")  # Add spacing between results

        return "\n".join(answer_parts)


def knowledge_assistant_node(state: GraphState) -> GraphState:
    """Knowledge Assistant node that answers questions from the knowledge base."""
    if not state["messages"]:
        return state

    todo_list = state.get("todo_list", [])
    if not todo_list:
        return state

    # Get the next task for knowledge assistant
    next_task = get_next_task(todo_list)
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
    answer = assistant.generate_answer(query)

    new_state = state.copy()
    msgs = list(new_state["messages"])
    msgs.append(AIMessage(content=f"📚 Knowledge Assistant: {answer}"))
    msgs.append(AIMessage(content=f"📋 Task: {next_task['description']}"))

    # Mark task as completed
    todo_list = mark_task_completed(todo_list, next_task["id"], answer)
    new_state["todo_list"] = todo_list

    new_state["messages"] = msgs
    return new_state
