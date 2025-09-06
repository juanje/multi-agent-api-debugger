"""
Mock routing intelligence for the Supervisor agent.

This module contains all the regex patterns and routing logic that would
normally be handled by an LLM in a real implementation.
"""

import re
from typing import List, Optional
from multi_agent.state import GraphState

# Regex patterns for routing decisions
API_PATTERNS = [
    r"\b(run|start|execute|submit)\s+(job|task)\b",
    r"\b(get|fetch|retrieve)\s+(job|result|status)\b",
    r"\b(list|show)\s+.*\s+(jobs|tasks)\b",
    r"\b(check|monitor)\s+(system|status)\b",
    r"\b(api|endpoint|call)\b",
]

DEBUG_PATTERNS = [
    r"\b(debug|analyze|investigate|troubleshoot)\b",
    r"\b(error|failed|failure|problem)\b",
    r"\b(why|what went wrong|root cause)\b",
    r"\b(logs|diagnose|fix)\b",
]

KNOWLEDGE_PATTERNS = [
    r"\b(what|how|when|where|why)\b.*\?",
    r"\b(explain|describe|tell me about)\b",
    r"\b(help|documentation|manual|guide)\b",
    # Only match API-related questions, not commands
    r"\b(what is|what are|how does|how do)\b.*\b(api|jobs|authentication|templates)\b",
]

RESPONSE_PATTERNS = [
    r"\b(format|summarize|finalize|complete)\b",
    r"\b(done|finished|ready)\b",
]


def matches_patterns(text: str, patterns: List[str]) -> bool:
    """Check if text matches any of the given patterns."""
    text_lower = text.lower()
    return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in patterns)


def extract_api_operations(text: str) -> List[str]:
    """Extract API operations from text (mock LLM extraction)."""
    operations = []
    text_lower = text.lower()

    if "list" in text_lower and ("job" in text_lower or "task" in text_lower):
        operations.append("list_public_jobs")
    elif "run" in text_lower and "job" in text_lower:
        # Extract job name
        job_name = extract_job_name(text)
        operations.append(f"run_job:job_name={job_name}")
    elif "get" in text_lower and ("result" in text_lower or "status" in text_lower):
        # Extract job ID
        job_id = extract_job_id(text)
        if job_id:
            operations.append(f"get_job_results:job_id={job_id}")
    elif "check" in text_lower and "system" in text_lower:
        operations.append("check_system_status")

    return operations


def extract_job_name(text: str) -> str:
    """Extract job name from text (mock LLM extraction)."""
    text_lower = text.lower()

    # Common job names
    if "data" in text_lower and "process" in text_lower:
        return "data_processing"
    elif "image" in text_lower and "analysis" in text_lower:
        return "image_analysis"
    elif "report" in text_lower and "generation" in text_lower:
        return "report_generation"
    elif "validation" in text_lower:
        return "data_validation"
    else:
        return "data_processing"  # Default


def extract_job_id(text: str) -> Optional[str]:
    """Extract job ID from text (mock LLM extraction)."""
    # Look for job_XXX pattern
    job_id_match = re.search(r"job_(\d{3})", text.lower())
    if job_id_match:
        return f"job_{job_id_match.group(1)}"
    return None


def determine_route(state: GraphState) -> str:
    """Determine the next route based on current state (mock LLM decision)."""
    # Check if we're done
    if state.get("final_response"):
        return "done"

    # Check if we have results or analysis to synthesize
    if state.get("results") or state.get("root_cause_analysis"):
        return "response_synthesizer"

    # Check if there are pending todo items
    todo_list = state.get("todo_list", [])
    if todo_list:
        return "api_operator"

    # Check if there's error info to analyze (but only if we haven't analyzed it yet)
    if state.get("error_info") and not state.get("root_cause_analysis"):
        return "debugger"

    # Otherwise, route based on message content
    if not state.get("messages"):
        return "done"

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

    # Route based on content patterns
    # Prioritize questions and knowledge requests over API operations
    if matches_patterns(text, DEBUG_PATTERNS):
        return "debugger"
    elif matches_patterns(text, KNOWLEDGE_PATTERNS):
        return "knowledge_assistant"
    elif matches_patterns(text, API_PATTERNS):
        return "api_operator"
    elif text.strip():  # If there's actual content, default to API operations
        return "api_operator"
    else:  # Empty content, we're done
        return "done"


def create_todo_list(state: GraphState) -> List[str]:
    """Create todo list based on user request (mock LLM planning)."""
    if not state.get("messages"):
        return []

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

    # Extract operations from text
    operations = extract_api_operations(text)

    # If no operations found, default to listing jobs
    if not operations:
        operations = ["list_public_jobs"]

    return operations
