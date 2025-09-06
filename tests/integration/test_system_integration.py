"""
Integration tests for the multi-agent system.
"""

from langchain_core.messages import HumanMessage

from multi_agent.graph import app


def test_tools_calculation():
    """Test 1: Tools - calculation."""
    state = {
        "messages": [HumanMessage(content="What is 12*7?")],
        "route": None,
        "operations": None,
        "steps": None,
        "results": None,
        "current_step": None,
        "pending_tasks": None,
    }
    for _ in app.stream(state, config={"configurable": {"thread_id": "test1"}}):
        pass
    final = app.get_state({"configurable": {"thread_id": "test1"}}).values
    assert final["route"] == "tools"
    assert "12*7 = 84" in final["messages"][-1].content


def test_rag_document():
    """Test 2: RAG - document."""
    state = {
        "messages": [HumanMessage(content="Summarize the product manual (chapters).")],
        "route": None,
        "operations": None,
        "steps": None,
        "results": None,
        "current_step": None,
        "pending_tasks": None,
    }
    for _ in app.stream(state, config={"configurable": {"thread_id": "test2"}}):
        pass
    final = app.get_state({"configurable": {"thread_id": "test2"}}).values
    assert final["route"] == "rag"
    assert "(RAG-mock)" in final["messages"][-1].content


def test_tools_time():
    """Test 3: Tools - time."""
    state = {
        "messages": [HumanMessage(content="What time is it right now?")],
        "route": None,
        "operations": None,
        "steps": None,
        "results": None,
        "current_step": None,
        "pending_tasks": None,
    }
    for _ in app.stream(state, config={"configurable": {"thread_id": "test3"}}):
        pass
    final = app.get_state({"configurable": {"thread_id": "test3"}}).values
    assert final["route"] == "tools"
    assert "Current time (mock):" in final["messages"][-1].content


def test_tools_ambiguous():
    """Test 4: Tools - ambiguous without '?'."""
    state = {
        "messages": [HumanMessage(content="hello")],
        "route": None,
        "operations": None,
        "steps": None,
        "results": None,
        "current_step": None,
        "pending_tasks": None,
    }
    for _ in app.stream(state, config={"configurable": {"thread_id": "test4"}}):
        pass
    final = app.get_state({"configurable": {"thread_id": "test4"}}).values
    assert final["route"] == "tools"
    assert "mock tools" in final["messages"][-1].content


def test_rag_ambiguous():
    """Test 5: RAG - ambiguous with '?'."""
    state = {
        "messages": [HumanMessage(content="Can you help me?")],
        "route": None,
        "operations": None,
        "steps": None,
        "results": None,
        "current_step": None,
        "pending_tasks": None,
    }
    for _ in app.stream(state, config={"configurable": {"thread_id": "test5"}}):
        pass
    final = app.get_state({"configurable": {"thread_id": "test5"}}).values
    assert final["route"] == "rag"
    assert "(RAG-mock)" in final["messages"][-1].content


def test_parallel_operations():
    """Test 6: Parallel - two calculations."""
    state = {
        "messages": [HumanMessage(content="What is 2+3 and 8/2?")],
        "route": None,
        "operations": None,
        "steps": None,
        "results": None,
        "current_step": None,
        "pending_tasks": None,
    }
    for _ in app.stream(state, config={"configurable": {"thread_id": "test6"}}):
        pass
    final = app.get_state({"configurable": {"thread_id": "test6"}}).values
    assert final["route"] == "parallel"
    # Verify there are two messages: intermediate and final
    assert len(final["messages"]) >= 2
    assert "Executing operations in parallel" in final["messages"][-2].content
    assert "5 and 4.0" in final["messages"][-1].content


def test_sequential_operations():
    """Test 7: Sequential - sum of calculations."""
    state = {
        "messages": [HumanMessage(content="sum of 3x8 and 129/3")],
        "route": None,
        "operations": None,
        "steps": None,
        "results": None,
        "current_step": None,
        "pending_tasks": None,
    }
    for _ in app.stream(state, config={"configurable": {"thread_id": "test7"}}):
        pass
    final = app.get_state({"configurable": {"thread_id": "test7"}}).values
    assert final["route"] == "done"  # Now ends in "done"
    # Verify there are multiple messages (one for each step)
    assert len(final["messages"]) >= 4  # At least 4 messages (3 steps + final result)
    assert "Total sum" in final["messages"][-1].content


def run_integration_tests():
    """Runs all integration tests."""
    print("🧪 Running integration tests...")

    try:
        test_tools_calculation()
        test_rag_document()
        test_tools_time()
        test_tools_ambiguous()
        test_rag_ambiguous()
        test_parallel_operations()
        test_sequential_operations()

        print("✅ All integration tests passed successfully!")
        return True
    except Exception as e:
        print(f"❌ Error in integration tests: {e}")
        return False
