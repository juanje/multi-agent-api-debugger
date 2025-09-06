"""
Mock intelligence functions that simulate LLM capabilities.

This module contains functions that simulate what would normally be
handled by LLMs in a real implementation, such as:
- Text analysis and understanding
- Decision making
- Content generation
- Pattern recognition
"""

from typing import Dict, Any, List, cast


def analyze_user_intent(message: str) -> Dict[str, Any]:
    """Analyze user intent from message (mock LLM analysis)."""
    message_lower = message.lower()

    intent = {
        "primary_intent": "unknown",
        "confidence": 0.0,
        "entities": [],
        "sentiment": "neutral",
    }

    # Determine primary intent
    if any(word in message_lower for word in ["list", "show", "get"]):
        intent["primary_intent"] = "query"
        intent["confidence"] = 0.9
    elif any(word in message_lower for word in ["run", "start", "execute"]):
        intent["primary_intent"] = "action"
        intent["confidence"] = 0.9
    elif any(word in message_lower for word in ["debug", "analyze", "error"]):
        intent["primary_intent"] = "debug"
        intent["confidence"] = 0.9
    elif "?" in message:
        intent["primary_intent"] = "question"
        intent["confidence"] = 0.8

    # Extract entities
    entities = cast(List[Dict[str, str]], intent.get("entities") or [])
    if "job" in message_lower:
        entities.append({"type": "resource", "value": "job"})
    if "api" in message_lower:
        entities.append({"type": "resource", "value": "api"})
    if "system" in message_lower:
        entities.append({"type": "resource", "value": "system"})
    intent["entities"] = entities

    # Determine sentiment
    if any(word in message_lower for word in ["error", "failed", "problem", "issue"]):
        intent["sentiment"] = "negative"
    elif any(word in message_lower for word in ["help", "please", "thanks"]):
        intent["sentiment"] = "positive"

    return intent


def generate_agent_instruction(agent_type: str, context: Dict[str, Any]) -> str:
    """Generate instruction for agent (mock LLM generation)."""
    if agent_type == "api_operator":
        if "todo_list" in context and context["todo_list"]:
            return f"Execute: {context['todo_list'][0]}"
        return "List available jobs"

    elif agent_type == "debugger":
        if "error_info" in context and context["error_info"]:
            return (
                f"Analyze error: {context['error_info'].get('error_code', 'Unknown')}"
            )
        return "Debug the most recent error"

    elif agent_type == "knowledge_assistant":
        if "messages" in context and context["messages"]:
            last_msg = context["messages"][-1]
            if hasattr(last_msg, "content"):
                return f"Answer: {last_msg.content}"
        return "Provide general help"

    elif agent_type == "response_synthesizer":
        return "Format and present the final response"

    return "Process the current state"


def determine_confidence_level(analysis: Dict[str, Any]) -> str:
    """Determine confidence level for analysis (mock LLM assessment)."""
    confidence_score = analysis.get("confidence", 0.0)

    if confidence_score >= 0.8:
        return "High"
    elif confidence_score >= 0.6:
        return "Medium"
    else:
        return "Low"


def should_continue_processing(state: Dict[str, Any]) -> bool:
    """Determine if processing should continue (mock LLM decision)."""
    # Check if we have a final response
    if state.get("final_response"):
        return False

    # Check if we have pending todos
    if state.get("todo_list"):
        return True

    # Check if we have unresolved errors
    if state.get("error_info") and not state.get("root_cause_analysis"):
        return True

    # Check if we have results to synthesize
    if state.get("results") and not state.get("final_response"):
        return True

    return False


def extract_key_information(text: str, context: str) -> Dict[str, Any]:
    """Extract key information from text (mock LLM extraction)."""
    info = {"entities": [], "actions": [], "parameters": {}, "confidence": 0.7}

    text_lower = text.lower()

    # Extract entities
    entities = cast(List[str], info.get("entities") or [])
    if "job" in text_lower:
        entities.append("job")
    if "api" in text_lower:
        entities.append("api")
    if "error" in text_lower:
        entities.append("error")
    info["entities"] = entities

    # Extract actions
    actions = cast(List[str], info.get("actions") or [])
    if any(word in text_lower for word in ["run", "start", "execute"]):
        actions.append("execute")
    if any(word in text_lower for word in ["list", "show", "get"]):
        actions.append("query")
    if any(word in text_lower for word in ["debug", "analyze"]):
        actions.append("debug")
    info["actions"] = actions

    # Extract parameters
    parameters = cast(Dict[str, str], info.get("parameters") or {})
    if "job_" in text_lower:
        import re

        job_match = re.search(r"job_(\d{3})", text_lower)
        if job_match:
            parameters["job_id"] = f"job_{job_match.group(1)}"
    info["parameters"] = parameters

    return info


def generate_summary(content: str, max_length: int = 100) -> str:
    """Generate summary of content (mock LLM summarization)."""
    if len(content) <= max_length:
        return content

    # Simple truncation with ellipsis (in real implementation, this would be LLM-based)
    return content[: max_length - 3] + "..."


def classify_message_type(message: str) -> str:
    """Classify message type (mock LLM classification)."""
    message_lower = message.lower()

    if "?" in message:
        return "question"
    elif any(word in message_lower for word in ["run", "start", "execute", "submit"]):
        return "command"
    elif any(word in message_lower for word in ["error", "failed", "problem"]):
        return "error_report"
    elif any(word in message_lower for word in ["help", "how", "what", "explain"]):
        return "help_request"
    else:
        return "statement"
