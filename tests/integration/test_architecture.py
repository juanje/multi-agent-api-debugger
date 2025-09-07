"""
Integration tests for the new multi-agent architecture.
"""

import pytest
from langchain_core.messages import HumanMessage
from multi_agent.graph import get_graph


@pytest.mark.asyncio
async def test_api_operations_flow():
    """Test the complete API operations flow."""
    graph = get_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="list all jobs")]},
        config={"configurable": {"thread_id": "test-api"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"
    assert "results" in result
    assert "list_public_jobs" in result["results"]


@pytest.mark.asyncio
async def test_knowledge_query_flow():
    """Test the knowledge assistant flow."""
    graph = get_graph()
    result = await graph.ainvoke(
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


@pytest.mark.asyncio
async def test_debugging_flow():
    """Test the debugging flow."""
    graph = get_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="debug job_003")]},
        config={"configurable": {"thread_id": "test-debug"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"
    # Check that debugging flow completed successfully
    # Note: The actual debugging behavior may vary based on LLM decisions
    # But we should get a substantive response
    ai_messages = [
        msg for msg in result["messages"] if hasattr(msg, "type") and msg.type == "ai"
    ]
    assert len(ai_messages) > 0
    # Should contain debugging-related content
    debug_content = " ".join(str(msg.content) for msg in ai_messages).lower()
    assert any(
        word in debug_content for word in ["debug", "job_003", "analysis", "findings"]
    )


@pytest.mark.asyncio
async def test_run_job_flow():
    """Test running a job."""
    graph = get_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="run job data_processing")]},
        config={"configurable": {"thread_id": "test-run-job"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"
    # Check that we got some results (the LLM may interpret the request differently)
    # But should have executed some operation
    assert "results" in result or any(
        "job" in str(msg.content).lower()
        for msg in result["messages"]
        if hasattr(msg, "type") and msg.type == "ai"
    )
    # Should contain job-related response
    ai_messages = [
        msg for msg in result["messages"] if hasattr(msg, "type") and msg.type == "ai"
    ]
    job_content = " ".join(str(msg.content) for msg in ai_messages).lower()
    assert any(
        word in job_content for word in ["job", "data_processing", "run", "execute"]
    )


@pytest.mark.asyncio
async def test_system_status_flow():
    """Test checking system status."""
    graph = get_graph()
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="check system status")]},
        config={"configurable": {"thread_id": "test-status"}},
    )

    assert "messages" in result
    assert len(result["messages"]) > 1
    assert result.get("route") == "done"
    assert "results" in result
    assert "check_system_status" in result["results"]
