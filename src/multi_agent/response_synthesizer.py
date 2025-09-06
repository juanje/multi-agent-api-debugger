"""
Response Synthesizer agent for formatting user-facing responses.

This agent is responsible for synthesizing and formatting the final
response that will be presented to the user.
"""

from __future__ import annotations
from typing import Dict, Any, List
from langchain_core.messages import AIMessage
from multi_agent.state import GraphState
from multi_agent.mocks.data import get_response_template
from multi_agent.mocks.planning import get_next_task, mark_task_completed


class ResponseSynthesizer:
    """Response Synthesizer for formatting user-facing responses."""

    def __init__(self):
        """Initialize the Response Synthesizer."""
        pass

    def synthesize_response(self, state: GraphState) -> str:
        """Synthesize a response based on the current state."""
        # Check what type of response to generate
        if state.get("root_cause_analysis"):
            return self._format_debugging_response(state)
        elif state.get("error_info"):
            return self._format_api_error_response(state)
        elif state.get("results"):
            return self._format_api_success_response(state)
        elif self._has_knowledge_response(state):
            return self._format_knowledge_response(state)
        else:
            return self._format_general_response(state)

    def _format_api_success_response(self, state: GraphState) -> str:
        """Format response for successful API operations."""
        results = state.get("results") or {}

        # Create summary
        operations = list(results.keys())
        summary = f"Successfully executed {len(operations)} API operation(s): {', '.join(operations)}"

        # Create details
        details_parts = []
        for op, result in results.items():
            if isinstance(result, dict) and "error" not in result:
                details_parts.append(f"• {op}: {self._extract_key_info(result)}")

        details = "\n".join(details_parts) if details_parts else "No details available"

        template = get_response_template("api_success")
        return template["format"].format(
            title=template["title"], summary=summary, details=details
        )

    def _format_api_error_response(self, state: GraphState) -> str:
        """Format response for API errors."""
        error_info = state.get("error_info") or {}
        results = state.get("results") or {}

        # Create summary
        operations = list(results.keys())
        summary = f"API operation(s) failed: {', '.join(operations)}"

        # Create error details
        error_details = f"Error: {error_info.get('error', 'Unknown error')}"
        if error_info.get("error_code"):
            error_details += f"\nError Code: {error_info['error_code']}"

        # Create next steps
        next_steps = [
            "Check the error details above",
            "Verify your request parameters",
            "Try again with corrected parameters",
            "Contact support if the issue persists",
        ]

        template = get_response_template("api_error")
        return template["format"].format(
            title=template["title"],
            summary=summary,
            error_details=error_details,
            next_steps="\n".join(f"• {step}" for step in next_steps),
        )

    def _format_debugging_response(self, state: GraphState) -> str:
        """Format response for debugging analysis."""
        analysis = state.get("root_cause_analysis") or {}

        # Create analysis summary
        analysis_summary = f"Analyzed error: {analysis.get('error_code', 'Unknown')} with {analysis.get('confidence_level', 'Unknown')} confidence"

        # Create root cause
        root_cause = analysis.get("root_cause_hypothesis", "No root cause identified")

        # Create recommended actions
        recommended_actions = analysis.get("recommended_actions", [])
        if not recommended_actions:
            recommended_actions = ["No specific recommendations available"]

        template = get_response_template("debugging_complete")
        return template["format"].format(
            title=template["title"],
            analysis_summary=analysis_summary,
            root_cause=root_cause,
            recommended_actions="\n".join(
                f"• {action}" for action in recommended_actions
            ),
        )

    def _has_knowledge_response(self, state: GraphState) -> bool:
        """Check if there's a knowledge assistant response in the messages."""
        messages = state.get("messages", [])
        for msg in messages:
            if hasattr(msg, "content") and "📚 Knowledge Assistant:" in str(
                msg.content
            ):
                return True
        return False

    def _format_knowledge_response(self, state: GraphState) -> str:
        """Format response for knowledge assistant queries."""
        messages = state.get("messages", [])
        knowledge_response = None

        # Find the knowledge assistant response
        for msg in messages:
            if hasattr(msg, "content") and "📚 Knowledge Assistant:" in str(
                msg.content
            ):
                knowledge_response = str(msg.content)
                break

        if not knowledge_response:
            return self._format_general_response(state)

        # Extract the actual response content (remove the emoji prefix)
        if "📚 Knowledge Assistant:" in knowledge_response:
            content = knowledge_response.split("📚 Knowledge Assistant:", 1)[1].strip()
        else:
            content = knowledge_response

        template = get_response_template("knowledge_response")
        return template["format"].format(title=template["title"], answer=content)

    def _format_general_response(self, state: GraphState) -> str:
        """Format general response."""
        content = "Operation completed successfully"
        template = get_response_template("general")
        return template["format"].format(title=template["title"], content=content)

    def _extract_key_info(self, result: Dict[str, Any]) -> str:
        """Extract key information from a result dictionary."""
        if "job_id" in result:
            return f"Job ID: {result['job_id']}"
        elif "status" in result:
            return f"Status: {result['status']}"
        elif "message" in result:
            return result["message"]
        else:
            return str(result)

    def create_knowledge_summary(self, state: GraphState) -> List[Dict[str, Any]]:
        """Create a knowledge summary for potential LTM storage."""
        summary = []

        # Add API operations to summary
        results = state.get("results") or {}
        if results:
            summary.append(
                {
                    "type": "api_operations",
                    "operations_performed": list(results.keys()),
                    "success": not any(
                        "error" in str(result) for result in results.values()
                    ),
                    "timestamp": "2024-01-15T10:00:00Z",
                }
            )

        # Add debugging analysis to summary
        analysis = state.get("root_cause_analysis")
        if analysis:
            summary.append(
                {
                    "type": "debugging_analysis",
                    "error_code": analysis.get("error_code"),
                    "confidence_level": analysis.get("confidence_level"),
                    "severity": analysis.get("severity"),
                    "timestamp": "2024-01-15T10:00:00Z",
                }
            )

        return summary


def response_synthesizer_node(state: GraphState) -> GraphState:
    """Response Synthesizer node that formats the final response."""
    if not state["messages"]:
        return state

    todo_list = state.get("todo_list") or []

    # Get the next task for response synthesizer (if any)
    next_task = get_next_task(todo_list)
    if next_task and next_task["agent"] == "response_synthesizer":
        # This is a specific synthesis task
        msgs = list(state["messages"])
        msgs.append(AIMessage(content=f"📋 Task: {next_task['description']}"))

        # Mark task as completed
        todo_list = mark_task_completed(
            todo_list, next_task["id"], "Response synthesized"
        )
    else:
        # General response synthesis
        msgs = list(state["messages"])

    synthesizer = ResponseSynthesizer()
    final_response = synthesizer.synthesize_response(state)
    knowledge_summary = synthesizer.create_knowledge_summary(state)

    new_state = state.copy()
    new_state["final_response"] = final_response
    new_state["knowledge_summary"] = knowledge_summary
    new_state["route"] = "done"
    new_state["todo_list"] = todo_list

    # Add final response message
    msgs.append(AIMessage(content=final_response))
    new_state["messages"] = msgs

    return new_state
