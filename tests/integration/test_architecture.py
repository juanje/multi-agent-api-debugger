"""
Integration tests for the new multi-agent architecture.
"""

from langchain_core.messages import HumanMessage
from multi_agent.graph import app


def test_api_operations_flow():
    """Test the complete API operations flow."""
    result = app.invoke(
        {"messages": [HumanMessage(content="list all jobs")]},
        config={"configurable": {"thread_id": "test-api"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"
    assert "results" in result
    assert "list_public_jobs" in result["results"]


def test_knowledge_query_flow():
    """Test the knowledge assistant flow."""
    result = app.invoke(
        {"messages": [HumanMessage(content="what is the API?")]},
        config={"configurable": {"thread_id": "test-knowledge"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"

    # Check that we got a knowledge response
    ai_messages = [
        msg for msg in result["messages"] if hasattr(msg, "type") and msg.type == "ai"
    ]
    assert len(ai_messages) > 0
    assert any("API" in str(msg.content) for msg in ai_messages)


def test_debugging_flow():
    """Test the debugging flow."""
    result = app.invoke(
        {"messages": [HumanMessage(content="debug job_003")]},
        config={"configurable": {"thread_id": "test-debug"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"
    assert "error_info" in result
    assert "root_cause_analysis" in result
    assert result["error_info"]["error_code"] == "TEMPLATE_NOT_FOUND"


def test_run_job_flow():
    """Test running a job."""
    result = app.invoke(
        {"messages": [HumanMessage(content="run job data_processing")]},
        config={"configurable": {"thread_id": "test-run-job"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"
    assert "results" in result
    assert "run_job" in result["results"]


def test_system_status_flow():
    """Test checking system status."""
    result = app.invoke(
        {"messages": [HumanMessage(content="check system status")]},
        config={"configurable": {"thread_id": "test-status"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"
    assert "results" in result
    assert "check_system_status" in result["results"]
