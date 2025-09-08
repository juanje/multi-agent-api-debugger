"""
Debugger agent for analyzing errors and providing root cause analysis.

This agent is responsible for analyzing errors and providing structured
root cause analysis with recommended actions.
"""

from __future__ import annotations
from typing import Dict, Any
from langchain_core.messages import AIMessage
from ..graph.state import GraphState
from ..graph.planning import get_next_task
from ..utils.mocks.data import get_error_pattern
from ..graph.planning import mark_task_completed, mark_task_failed
from ..memory import get_ltm_service, LTM_CONFIDENCE_THRESHOLD
from ..llm import should_use_mocks


class Debugger:
    """Debugger for analyzing errors and providing root cause analysis."""

    def __init__(self):
        """Initialize the Debugger."""
        pass

    async def analyze_error(self, error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze error and provide root cause analysis."""
        error_code = error_info.get("error_code", "UNKNOWN_ERROR")
        analysis = get_error_pattern(error_code)

        # Enhance analysis with similar cases from LTM (only in production mode)
        if not should_use_mocks():
            try:
                ltm_service = get_ltm_service()
                error_message = error_info.get("error_message", "")
                similar_errors = await ltm_service.search_similar_errors(
                    error_code, error_message, limit=3
                )

                if similar_errors:
                    # Add historical context to analysis
                    historical_insights = []
                    for result in similar_errors:
                        memory = result.memory
                        confidence = memory.confidence_level or "Unknown"
                        severity = memory.severity or "Unknown"
                        historical_insights.append(
                            {
                                "confidence": confidence,
                                "severity": severity,
                                "similarity_score": round(result.similarity_score, 2),
                                "timestamp": memory.timestamp,
                            }
                        )

                    analysis["historical_cases"] = historical_insights
                    analysis["enhanced_with_ltm"] = True

                    # Update confidence if we have high-confidence historical cases
                    high_confidence_cases = [
                        case
                        for case in historical_insights
                        if case["confidence"] == "High"
                        and isinstance(case["similarity_score"], (int, float))
                        and case["similarity_score"] > LTM_CONFIDENCE_THRESHOLD
                    ]
                    if high_confidence_cases:
                        analysis["confidence_level"] = (
                            "High (confirmed by historical cases)"
                        )

            except Exception as e:
                # Don't fail analysis if LTM lookup fails
                analysis["ltm_search_error"] = str(e)

        return analysis


async def debugger_node(state: GraphState) -> GraphState:
    """Debugger node that analyzes errors and provides root cause analysis."""
    if not state["messages"]:
        # No messages - set route to response_synthesizer
        new_state = state.copy()
        new_state["route"] = "response_synthesizer"
        return new_state

    todo_list = state.get("todo_list", [])
    if not todo_list:
        # No todo list - create a debugging task and go to response_synthesizer
        new_state = state.copy()
        new_state["route"] = "response_synthesizer"
        return new_state

    # Get the next task for debugger
    next_task = await get_next_task(todo_list)
    if not next_task or next_task["agent"] != "debugger":
        # No debugger task - go to response_synthesizer
        new_state = state.copy()
        new_state["route"] = "response_synthesizer"
        return new_state

    # Check if there's error information to analyze
    error_info = state.get("error_info")

    if not error_info:
        # If no error_info but debugging was requested, create mock error
        error_type = next_task["parameters"].get("error_type", "template_not_found")

        if error_type == "template_not_found":
            error_info = {
                "error_code": "TEMPLATE_NOT_FOUND",
                "error_message": "Template 'standard' not found in system",
                "timestamp": "2024-01-15T10:35:00Z",
                "suggestions": ["Use 'basic' template", "Check template availability"],
            }
        else:
            # No error to analyze, mark task as failed
            todo_list = mark_task_failed(
                todo_list, next_task["id"], "No error information available"
            )
            new_state = state.copy()
            new_state["todo_list"] = todo_list
            new_state["route"] = "response_synthesizer"
            return new_state

    # Perform root cause analysis
    debugger = Debugger()
    analysis = await debugger.analyze_error(error_info)

    new_state = state.copy()
    msgs = list(new_state["messages"])
    msgs.append(AIMessage(content="🐞 Debugger analyzing error..."))
    msgs.append(AIMessage(content=f"📋 Task: {next_task['description']}"))

    # Store error_info and analysis in state
    new_state["error_info"] = error_info
    new_state["root_cause_analysis"] = analysis

    # Mark task as completed
    todo_list = mark_task_completed(todo_list, next_task["id"], analysis)
    new_state["todo_list"] = todo_list

    # Create analysis summary
    summary = f"""
🔍 Root Cause Analysis

Error: {analysis["error_code"]} - {analysis["error_message"]}
Hypothesis: {analysis["root_cause_hypothesis"]}
Confidence: {analysis["confidence_level"]}
Severity: {analysis["severity"]}

Recommended Actions:
{chr(10).join(f"• {action}" for action in analysis["recommended_actions"])}

Related Components: {", ".join(analysis["related_components"])}
"""

    msgs.append(AIMessage(content=summary))
    new_state["messages"] = msgs

    # Set next route - go to response_synthesizer to format final response
    new_state["route"] = "response_synthesizer"

    return new_state
