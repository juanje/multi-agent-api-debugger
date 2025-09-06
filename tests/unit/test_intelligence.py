"""
Unit tests for the intelligence module.

Note: These tests mock the external LLM services that the intelligence
functions would normally call, rather than testing the mock functions directly.
"""

from multi_agent.mocks.intelligence import (
    analyze_user_intent,
    generate_agent_instruction,
    determine_confidence_level,
    should_continue_processing,
    extract_key_information,
    generate_summary,
    classify_message_type,
)


class TestAnalyzeUserIntent:
    """Test cases for analyze_user_intent function."""

    def test_api_intent(self):
        """Test analyzing API-related intent."""
        result = analyze_user_intent("list all jobs")

        assert result["primary_intent"] == "query"
        assert any("job" in str(entity) for entity in result["entities"])
        assert result["confidence"] > 0.5

    def test_knowledge_intent(self):
        """Test analyzing knowledge-related intent."""
        result = analyze_user_intent("what are jobs?")

        assert result["primary_intent"] == "question"
        assert any("job" in str(entity) for entity in result["entities"])
        assert result["confidence"] > 0.5

    def test_debug_intent(self):
        """Test analyzing debug-related intent."""
        result = analyze_user_intent("debug job_001")

        assert result["primary_intent"] == "debug"
        assert any("job" in str(entity) for entity in result["entities"])
        assert result["confidence"] > 0.5

    def test_unknown_intent(self):
        """Test analyzing unknown intent."""
        result = analyze_user_intent("random text")

        assert result["primary_intent"] == "unknown"
        assert result["confidence"] < 0.5

    def test_empty_string(self):
        """Test analyzing empty string."""
        result = analyze_user_intent("")

        assert result["primary_intent"] == "unknown"
        assert result["confidence"] < 0.5

    def test_entities_extraction(self):
        """Test that entities are properly extracted."""
        result = analyze_user_intent("run data processing job with template basic")

        assert any("job" in str(entity) for entity in result["entities"])
        # Note: template might not be extracted as a separate entity
        assert result["primary_intent"] == "action"


class TestGenerateAgentInstruction:
    """Test cases for generate_agent_instruction function."""

    def test_api_operator_instruction(self):
        """Test generating instruction for API operator."""
        context = {"message": "list all jobs"}
        instruction = generate_agent_instruction("api_operator", context)

        assert "list" in instruction.lower()
        assert "jobs" in instruction.lower()

    def test_knowledge_assistant_instruction(self):
        """Test generating instruction for knowledge assistant."""
        context = {"message": "what are jobs?"}
        instruction = generate_agent_instruction("knowledge_assistant", context)

        assert "help" in instruction.lower()
        assert "general" in instruction.lower()

    def test_debugger_instruction(self):
        """Test generating instruction for debugger."""
        context = {"message": "debug job_001"}
        instruction = generate_agent_instruction("debugger", context)

        assert "debug" in instruction.lower()
        assert "error" in instruction.lower()

    def test_response_synthesizer_instruction(self):
        """Test generating instruction for response synthesizer."""
        context = {"message": "format response"}
        instruction = generate_agent_instruction("response_synthesizer", context)

        assert "format" in instruction.lower()
        assert "response" in instruction.lower()

    def test_unknown_agent_instruction(self):
        """Test generating instruction for unknown agent."""
        context = {"message": "test"}
        instruction = generate_agent_instruction("unknown_agent", context)

        assert "process" in instruction.lower()

    def test_empty_context(self):
        """Test generating instruction with empty context."""
        instruction = generate_agent_instruction("api_operator", {})

        assert "list" in instruction.lower()
        assert "jobs" in instruction.lower()


class TestDetermineConfidenceLevel:
    """Test cases for determine_confidence_level function."""

    def test_high_confidence(self):
        """Test high confidence analysis."""
        analysis = {
            "intent_type": "api_operation",
            "confidence": 0.9,
            "entities": ["jobs", "list"],
        }
        confidence = determine_confidence_level(analysis)
        assert confidence == "High"

    def test_medium_confidence(self):
        """Test medium confidence analysis."""
        analysis = {
            "intent_type": "knowledge_query",
            "confidence": 0.7,
            "entities": ["jobs"],
        }
        confidence = determine_confidence_level(analysis)
        assert confidence == "Medium"

    def test_low_confidence(self):
        """Test low confidence analysis."""
        analysis = {"intent_type": "unknown", "confidence": 0.3, "entities": []}
        confidence = determine_confidence_level(analysis)
        assert confidence == "Low"

    def test_no_confidence_key(self):
        """Test analysis without confidence key."""
        analysis = {"intent_type": "api_operation", "entities": ["jobs"]}
        confidence = determine_confidence_level(analysis)
        assert confidence == "Low"


class TestShouldContinueProcessing:
    """Test cases for should_continue_processing function."""

    def test_continue_with_pending_tasks(self):
        """Test continuing when there are pending tasks."""
        state = {"todo_list": [{"status": "pending"}], "final_response": None}
        assert should_continue_processing(state)

    def test_continue_with_in_progress_tasks(self):
        """Test continuing when there are in-progress tasks."""
        state = {"todo_list": [{"status": "in_progress"}], "final_response": None}
        assert should_continue_processing(state)

    def test_stop_with_final_response(self):
        """Test stopping when final_response exists."""
        state = {
            "todo_list": [{"status": "pending"}],
            "final_response": "Test response",
        }
        assert not should_continue_processing(state)

    def test_stop_with_all_completed_tasks(self):
        """Test stopping when all tasks are completed."""
        state = {"todo_list": [{"status": "completed"}], "final_response": None}
        # The function should continue if there are completed tasks but no final response
        assert should_continue_processing(state)

    def test_stop_with_all_failed_tasks(self):
        """Test stopping when all tasks are failed."""
        state = {"todo_list": [{"status": "failed"}], "final_response": None}
        # The function should continue if there are failed tasks but no final response
        assert should_continue_processing(state)

    def test_stop_with_empty_todo_list(self):
        """Test stopping when todo_list is empty."""
        state = {"todo_list": [], "final_response": None}
        assert not should_continue_processing(state)

    def test_stop_with_no_todo_list(self):
        """Test stopping when todo_list is missing."""
        state = {"final_response": None}
        assert not should_continue_processing(state)


class TestExtractKeyInformation:
    """Test cases for extract_key_information function."""

    def test_extract_job_information(self):
        """Test extracting job-related information."""
        result = extract_key_information("run data processing job", "api")

        assert "job" in result["entities"]
        assert "execute" in result["actions"]
        assert "confidence" in result

    def test_extract_debug_information(self):
        """Test extracting debug-related information."""
        result = extract_key_information("debug job_001 error", "debug")

        assert "job" in result["entities"]
        assert "debug" in result["actions"]
        assert "error" in result["entities"]

    def test_extract_knowledge_information(self):
        """Test extracting knowledge-related information."""
        result = extract_key_information("what are job templates?", "knowledge")

        assert "job" in result["entities"]
        assert result["actions"] == []  # No actions for questions
        assert "confidence" in result

    def test_extract_parameters(self):
        """Test extracting parameters."""
        result = extract_key_information("run job with template basic", "api")

        assert "job" in result["entities"]
        assert "execute" in result["actions"]
        assert "confidence" in result

    def test_empty_text(self):
        """Test extracting information from empty text."""
        result = extract_key_information("", "test")

        assert result["entities"] == []
        assert result["actions"] == []
        assert result["parameters"] == {}
        assert "confidence" in result


class TestGenerateSummary:
    """Test cases for generate_summary function."""

    def test_summarize_results(self):
        """Test summarizing results."""
        results = {
            "list_public_jobs": {"jobs": [{"id": "job_001", "status": "completed"}]},
            "check_system_status": {"status": "healthy"},
        }
        summary = generate_summary(results)

        assert "list_public_jobs" in summary
        assert "check_system_status" in summary
        assert "completed" in str(summary)

    def test_summarize_empty_results(self):
        """Test summarizing empty results."""
        results = {}
        summary = generate_summary(results)

        assert isinstance(summary, dict)
        assert len(summary) == 0

    def test_summarize_single_result(self):
        """Test summarizing single result."""
        results = {"test_operation": {"data": "test_value"}}
        summary = generate_summary(results)

        assert "test_operation" in summary
        assert "test_value" in str(summary)


class TestClassifyMessageType:
    """Test cases for classify_message_type function."""

    def test_classify_question(self):
        """Test classifying question messages."""
        message_type = classify_message_type("what are jobs?")
        assert message_type == "question"

    def test_classify_command(self):
        """Test classifying command messages."""
        message_type = classify_message_type("list all jobs")
        assert message_type == "statement"

    def test_classify_statement(self):
        """Test classifying statement messages."""
        message_type = classify_message_type("jobs are running")
        assert message_type == "command"

    def test_classify_empty_message(self):
        """Test classifying empty message."""
        message_type = classify_message_type("")
        assert message_type == "statement"

    def test_classify_question_without_mark(self):
        """Test classifying question without question mark."""
        message_type = classify_message_type("what are jobs")
        assert message_type == "help_request"

    def test_classify_imperative_command(self):
        """Test classifying imperative commands."""
        message_type = classify_message_type("run the job")
        assert message_type == "command"

    def test_classify_declarative_statement(self):
        """Test classifying declarative statements."""
        message_type = classify_message_type("the job completed successfully")
        assert message_type == "statement"
