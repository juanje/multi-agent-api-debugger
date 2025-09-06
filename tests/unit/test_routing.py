"""
Unit tests for the routing module.
"""

from langchain_core.messages import HumanMessage
from multi_agent.mocks.routing import (
    matches_patterns,
    extract_job_id,
    determine_route,
    create_todo_list,
    API_PATTERNS,
    DEBUG_PATTERNS,
    KNOWLEDGE_PATTERNS,
    RESPONSE_PATTERNS,
)
from multi_agent.state import GraphState


class TestMatchesPatterns:
    """Test cases for matches_patterns function."""

    def test_matches_api_patterns(self):
        """Test matching API patterns."""
        # Test various API patterns
        assert matches_patterns("list all jobs", API_PATTERNS)
        assert matches_patterns("show all jobs", API_PATTERNS)
        assert matches_patterns("run job", API_PATTERNS)
        assert matches_patterns("execute task", API_PATTERNS)
        assert matches_patterns("get job results", API_PATTERNS)
        assert matches_patterns("fetch status", API_PATTERNS)
        assert matches_patterns("check system status", API_PATTERNS)
        assert matches_patterns("monitor status", API_PATTERNS)
        assert matches_patterns("api call", API_PATTERNS)
        assert matches_patterns("endpoint", API_PATTERNS)

    def test_matches_debug_patterns(self):
        """Test matching debug patterns."""
        assert matches_patterns("debug job_001", DEBUG_PATTERNS)
        assert matches_patterns("analyze error", DEBUG_PATTERNS)
        assert matches_patterns("investigate problem", DEBUG_PATTERNS)
        assert matches_patterns("fix error", DEBUG_PATTERNS)
        assert matches_patterns("troubleshoot", DEBUG_PATTERNS)
        assert matches_patterns("why did it fail", DEBUG_PATTERNS)
        assert matches_patterns("what went wrong", DEBUG_PATTERNS)
        assert matches_patterns("root cause", DEBUG_PATTERNS)
        assert matches_patterns("check logs", DEBUG_PATTERNS)
        assert matches_patterns("diagnose issue", DEBUG_PATTERNS)

    def test_matches_knowledge_patterns(self):
        """Test matching knowledge patterns."""
        assert matches_patterns("what are jobs?", KNOWLEDGE_PATTERNS)
        assert matches_patterns("how do I run a job?", KNOWLEDGE_PATTERNS)
        assert matches_patterns("when should I use templates?", KNOWLEDGE_PATTERNS)
        assert matches_patterns("where can I find documentation?", KNOWLEDGE_PATTERNS)
        assert matches_patterns("why did my job fail?", KNOWLEDGE_PATTERNS)
        assert matches_patterns("explain the API", KNOWLEDGE_PATTERNS)
        assert matches_patterns("describe authentication", KNOWLEDGE_PATTERNS)
        assert matches_patterns("tell me about jobs", KNOWLEDGE_PATTERNS)
        assert matches_patterns("help with templates", KNOWLEDGE_PATTERNS)
        assert matches_patterns("documentation for API", KNOWLEDGE_PATTERNS)
        assert matches_patterns("manual for jobs", KNOWLEDGE_PATTERNS)
        assert matches_patterns("guide for authentication", KNOWLEDGE_PATTERNS)

    def test_matches_response_patterns(self):
        """Test matching response patterns."""
        assert matches_patterns("format response", RESPONSE_PATTERNS)
        assert matches_patterns("summarize results", RESPONSE_PATTERNS)
        assert matches_patterns("finalize output", RESPONSE_PATTERNS)
        assert matches_patterns("complete task", RESPONSE_PATTERNS)
        assert matches_patterns("done", RESPONSE_PATTERNS)
        assert matches_patterns("finished", RESPONSE_PATTERNS)
        assert matches_patterns("ready", RESPONSE_PATTERNS)

    def test_no_matches(self):
        """Test text that doesn't match any patterns."""
        assert not matches_patterns("random text", API_PATTERNS)
        assert not matches_patterns("hello world", DEBUG_PATTERNS)
        assert not matches_patterns("goodbye", KNOWLEDGE_PATTERNS)
        assert not matches_patterns("test", RESPONSE_PATTERNS)

    def test_case_insensitive(self):
        """Test that matching is case insensitive."""
        assert matches_patterns("LIST ALL JOBS", API_PATTERNS)
        assert matches_patterns("DEBUG JOB_001", DEBUG_PATTERNS)
        assert matches_patterns("WHAT ARE JOBS?", KNOWLEDGE_PATTERNS)
        assert matches_patterns("DONE", RESPONSE_PATTERNS)


class TestExtractJobId:
    """Test cases for extract_job_id function."""

    def test_extract_valid_job_id(self):
        """Test extracting valid job ID."""
        assert extract_job_id("debug job_001") == "job_001"
        assert extract_job_id("analyze job_123") == "job_123"
        assert extract_job_id("fix job_999") == "job_999"

    def test_extract_job_id_case_insensitive(self):
        """Test extracting job ID case insensitive."""
        assert extract_job_id("DEBUG JOB_001") == "job_001"
        assert extract_job_id("Analyze Job_123") == "job_123"

    def test_extract_job_id_with_extra_text(self):
        """Test extracting job ID with extra text."""
        assert extract_job_id("please debug job_001 for me") == "job_001"
        assert extract_job_id("analyze the error in job_123") == "job_123"

    def test_no_job_id_found(self):
        """Test when no job ID is found."""
        assert extract_job_id("debug something") is None
        assert extract_job_id("analyze error") is None
        assert extract_job_id("fix problem") is None

    def test_invalid_job_id_format(self):
        """Test invalid job ID format."""
        assert extract_job_id("debug job_abc") is None
        assert extract_job_id("analyze job_12") is None  # Too short
        assert extract_job_id("fix job_1234") == "job_123"  # Captures first 3 digits


class TestDetermineRoute:
    """Test cases for determine_route function."""

    def test_done_when_final_response(self):
        """Test route is done when final_response exists."""
        state = GraphState(final_response="Test response")
        assert determine_route(state) == "done"

    def test_response_synthesizer_when_results(self):
        """Test route to response_synthesizer when results exist."""
        state = GraphState(results={"test": "data"})
        assert determine_route(state) == "response_synthesizer"

    def test_response_synthesizer_when_root_cause_analysis(self):
        """Test route to response_synthesizer when root_cause_analysis exists."""
        state = GraphState(root_cause_analysis={"test": "analysis"})
        assert determine_route(state) == "response_synthesizer"

    def test_api_operator_when_todo_list(self):
        """Test route to api_operator when todo_list exists."""
        state = GraphState(todo_list=[{"id": "task_001", "agent": "api_operator"}])
        assert determine_route(state) == "api_operator"

    def test_debugger_when_error_info(self):
        """Test route to debugger when error_info exists."""
        state = GraphState(
            error_info={"error_code": "TEST_ERROR"}, root_cause_analysis=None
        )
        assert determine_route(state) == "debugger"

    def test_api_operator_for_api_patterns(self):
        """Test route to api_operator for API patterns."""
        state = GraphState(messages=[HumanMessage(content="list all jobs")])
        assert determine_route(state) == "api_operator"

    def test_debugger_for_debug_patterns(self):
        """Test route to debugger for debug patterns."""
        state = GraphState(messages=[HumanMessage(content="debug job_001")])
        assert determine_route(state) == "debugger"

    def test_knowledge_assistant_for_knowledge_patterns(self):
        """Test route to knowledge_assistant for knowledge patterns."""
        state = GraphState(messages=[HumanMessage(content="what are jobs?")])
        assert determine_route(state) == "knowledge_assistant"

    def test_explain_without_question_mark(self):
        """Test explain commands without question mark go to knowledge_assistant."""
        state = GraphState(messages=[HumanMessage(content="explain job templates")])
        assert determine_route(state) == "knowledge_assistant"

    def test_default_to_api_operator(self):
        """Test default route to api_operator."""
        state = GraphState(messages=[HumanMessage(content="unknown command")])
        assert determine_route(state) == "api_operator"

    def test_empty_messages(self):
        """Test with empty messages."""
        state = GraphState(messages=[])
        assert determine_route(state) == "done"

    def test_list_content_message(self):
        """Test with list content in message."""
        message = HumanMessage(content=["list", "all", "jobs"])
        state = GraphState(messages=[message])
        assert determine_route(state) == "api_operator"

    def test_none_content_message(self):
        """Test with None content in message."""
        message = HumanMessage(content="")
        state = GraphState(messages=[message])
        assert determine_route(state) == "done"


class TestCreateTodoList:
    """Test cases for create_todo_list function."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = GraphState(messages=[])
        todo_list = create_todo_list(state)
        assert todo_list == []

    def test_list_jobs_command(self):
        """Test list jobs command."""
        state = GraphState(messages=[HumanMessage(content="list all jobs")])
        todo_list = create_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0] == "list_public_jobs"

    def test_run_job_command(self):
        """Test run job command."""
        state = GraphState(messages=[HumanMessage(content="run data processing job")])
        todo_list = create_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0] == "run_job:job_name=data_processing"

    def test_get_job_results_command(self):
        """Test get job results command."""
        state = GraphState(messages=[HumanMessage(content="get results for job_001")])
        todo_list = create_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0] == "get_job_results:job_id=job_001"

    def test_check_system_command(self):
        """Test check system command."""
        state = GraphState(messages=[HumanMessage(content="check system status")])
        todo_list = create_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0] == "check_system_status"

    def test_unknown_command_default(self):
        """Test unknown command defaults to list_public_jobs."""
        state = GraphState(messages=[HumanMessage(content="unknown command")])
        todo_list = create_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0] == "list_public_jobs"

    def test_list_content_message(self):
        """Test with list content in message."""
        message = HumanMessage(content=["list", "all", "jobs"])
        state = GraphState(messages=[message])
        todo_list = create_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0] == "list_public_jobs"

    def test_none_content_message(self):
        """Test with None content in message."""
        message = HumanMessage(content="")
        state = GraphState(messages=[message])
        todo_list = create_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0] == "list_public_jobs"
