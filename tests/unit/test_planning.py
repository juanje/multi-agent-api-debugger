"""
Unit tests for the planning module.
"""

from langchain_core.messages import HumanMessage
from multi_agent.mocks.planning import (
    create_task,
    create_comprehensive_todo_list,
    get_next_task,
    mark_task_completed,
    mark_task_failed,
    get_pending_tasks,
    get_completed_tasks,
    get_failed_tasks,
    is_workflow_complete,
    extract_job_name_from_text,
    extract_job_id_from_text,
)
from multi_agent.state import GraphState


class TestCreateTask:
    """Test cases for create_task function."""

    def test_create_basic_task(self):
        """Test creating a basic task."""
        task = create_task(description="Test task", agent="test_agent", priority=1)

        assert task["description"] == "Test task"
        assert task["agent"] == "test_agent"
        assert task["status"] == "pending"
        assert task["priority"] == 1
        assert task["dependencies"] == []
        assert task["parameters"] == {}
        assert task["result"] is None
        assert task["error"] is None
        assert "id" in task
        assert task["id"].startswith("task_")

    def test_create_task_with_dependencies(self):
        """Test creating a task with dependencies."""
        task = create_task(
            description="Test task",
            agent="test_agent",
            priority=2,
            dependencies=["task_001", "task_002"],
            parameters={"key": "value"},
        )

        assert task["dependencies"] == ["task_001", "task_002"]
        assert task["parameters"] == {"key": "value"}
        assert task["priority"] == 2

    def test_task_id_uniqueness(self):
        """Test that task IDs are unique."""
        task1 = create_task("Task 1", "agent1")
        task2 = create_task("Task 2", "agent2")

        assert task1["id"] != task2["id"]


class TestCreateComprehensiveTodoList:
    """Test cases for create_comprehensive_todo_list function."""

    def test_empty_messages(self):
        """Test with empty messages."""
        state = GraphState(messages=[])
        todo_list = create_comprehensive_todo_list(state)
        assert todo_list == []

    def test_list_jobs_command(self):
        """Test list jobs command."""
        state = GraphState(messages=[HumanMessage(content="list all jobs")])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0]["agent"] == "api_operator"
        assert "list" in todo_list[0]["description"].lower()
        assert todo_list[0]["parameters"]["operation"] == "list_public_jobs"

    def test_run_job_command(self):
        """Test run job command."""
        state = GraphState(messages=[HumanMessage(content="run data processing job")])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0]["agent"] == "api_operator"
        assert "execute" in todo_list[0]["description"].lower()
        assert todo_list[0]["parameters"]["operation"] == "run_job"

    def test_check_system_command(self):
        """Test check system command."""
        state = GraphState(messages=[HumanMessage(content="check system status")])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0]["agent"] == "api_operator"
        assert "status" in todo_list[0]["description"].lower()
        assert todo_list[0]["parameters"]["operation"] == "check_system_status"

    def test_knowledge_query(self):
        """Test knowledge query."""
        state = GraphState(messages=[HumanMessage(content="what are jobs?")])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0]["agent"] == "knowledge_assistant"
        assert "answer" in todo_list[0]["description"].lower()
        assert todo_list[0]["parameters"]["query"] == "what are jobs?"

    def test_explain_command(self):
        """Test explain command (without question mark)."""
        state = GraphState(messages=[HumanMessage(content="explain job templates")])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0]["agent"] == "knowledge_assistant"
        assert "answer" in todo_list[0]["description"].lower()

    def test_debug_command(self):
        """Test debug command."""
        state = GraphState(messages=[HumanMessage(content="debug job_003")])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 2  # Debug task + synthesis task
        assert todo_list[0]["agent"] == "debugger"
        assert todo_list[1]["agent"] == "response_synthesizer"
        assert todo_list[1]["dependencies"] == [todo_list[0]["id"]]

    def test_unknown_command_fallback(self):
        """Test unknown command fallback."""
        state = GraphState(messages=[HumanMessage(content="unknown command")])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0]["agent"] == "api_operator"
        assert "process" in todo_list[0]["description"].lower()

    def test_with_list_content(self):
        """Test with list content in message."""
        message = HumanMessage(content=["list", "all", "jobs"])
        state = GraphState(messages=[message])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0]["agent"] == "api_operator"

    def test_with_none_content(self):
        """Test with None content."""
        message = HumanMessage(content="")
        state = GraphState(messages=[message])
        todo_list = create_comprehensive_todo_list(state)

        assert len(todo_list) == 1
        assert todo_list[0]["agent"] == "api_operator"


class TestGetNextTask:
    """Test cases for get_next_task function."""

    def test_empty_todo_list(self):
        """Test with empty todo list."""
        result = get_next_task([])
        assert result is None

    def test_single_pending_task(self):
        """Test with single pending task."""
        task = create_task("Test task", "test_agent")
        todo_list = [task]
        result = get_next_task(todo_list)

        assert result == task

    def test_multiple_tasks_by_priority(self):
        """Test task selection by priority."""
        task1 = create_task("High priority", "agent1", priority=1)
        task2 = create_task("Low priority", "agent2", priority=3)
        todo_list = [task2, task1]  # Add in reverse order
        result = get_next_task(todo_list)

        assert result == task1  # Should select higher priority (lower number)

    def test_task_with_dependencies(self):
        """Test task with dependencies."""
        task1 = create_task("Task 1", "agent1", priority=1)
        task2 = create_task("Task 2", "agent2", priority=2, dependencies=[task1["id"]])
        todo_list = [task1, task2]

        # First call should return task1 (no dependencies)
        result1 = get_next_task(todo_list)
        assert result1 == task1

        # Mark task1 as completed
        todo_list = mark_task_completed(todo_list, task1["id"])

        # Second call should return task2 (dependencies satisfied)
        result2 = get_next_task(todo_list)
        assert result2 == task2

    def test_task_with_unsatisfied_dependencies(self):
        """Test task with unsatisfied dependencies."""
        task1 = create_task("Task 1", "agent1", priority=1)
        task2 = create_task("Task 2", "agent2", priority=2, dependencies=[task1["id"]])
        todo_list = [task2]  # Only task2, task1 not present
        result = get_next_task(todo_list)

        assert result is None  # No tasks ready due to missing dependency


class TestTaskStatusManagement:
    """Test cases for task status management functions."""

    def test_mark_task_completed(self):
        """Test marking task as completed."""
        task = create_task("Test task", "test_agent")
        todo_list = [task]
        result = mark_task_completed(todo_list, task["id"], "Success!")

        assert result[0]["status"] == "completed"
        assert result[0]["result"] == "Success!"

    def test_mark_task_failed(self):
        """Test marking task as failed."""
        task = create_task("Test task", "test_agent")
        todo_list = [task]
        result = mark_task_failed(todo_list, task["id"], "Error occurred")

        assert result[0]["status"] == "failed"
        assert result[0]["error"] == "Error occurred"

    def test_get_pending_tasks(self):
        """Test getting pending tasks."""
        task1 = create_task("Task 1", "agent1")
        task2 = create_task("Task 2", "agent2")
        todo_list = [task1, task2]
        todo_list = mark_task_completed(todo_list, task1["id"])

        pending = get_pending_tasks(todo_list)
        assert len(pending) == 1
        assert pending[0]["id"] == task2["id"]

    def test_get_completed_tasks(self):
        """Test getting completed tasks."""
        task1 = create_task("Task 1", "agent1")
        task2 = create_task("Task 2", "agent2")
        todo_list = [task1, task2]
        todo_list = mark_task_completed(todo_list, task1["id"])

        completed = get_completed_tasks(todo_list)
        assert len(completed) == 1
        assert completed[0]["id"] == task1["id"]

    def test_get_failed_tasks(self):
        """Test getting failed tasks."""
        task1 = create_task("Task 1", "agent1")
        task2 = create_task("Task 2", "agent2")
        todo_list = [task1, task2]
        todo_list = mark_task_failed(todo_list, task1["id"], "Error")

        failed = get_failed_tasks(todo_list)
        assert len(failed) == 1
        assert failed[0]["id"] == task1["id"]

    def test_is_workflow_complete(self):
        """Test workflow completion check."""
        task1 = create_task("Task 1", "agent1")
        task2 = create_task("Task 2", "agent2")
        todo_list = [task1, task2]

        # Not complete - tasks still pending
        assert not is_workflow_complete(todo_list)

        # Mark both as completed
        todo_list = mark_task_completed(todo_list, task1["id"])
        todo_list = mark_task_completed(todo_list, task2["id"])
        assert is_workflow_complete(todo_list)

        # Mark one as failed
        todo_list = mark_task_failed(todo_list, task1["id"], "Error")
        assert is_workflow_complete(todo_list)  # Still complete (completed or failed)

    def test_is_workflow_complete_empty(self):
        """Test workflow completion with empty list."""
        assert is_workflow_complete([])


class TestTextExtraction:
    """Test cases for text extraction functions."""

    def test_extract_job_name_data_processing(self):
        """Test extracting data processing job name."""
        result = extract_job_name_from_text("run data processing job")
        assert result == "data_processing"

    def test_extract_job_name_image_analysis(self):
        """Test extracting image analysis job name."""
        result = extract_job_name_from_text("execute image analysis task")
        assert result == "image_analysis"

    def test_extract_job_name_report_generation(self):
        """Test extracting report generation job name."""
        result = extract_job_name_from_text("start report generation")
        assert result == "report_generation"

    def test_extract_job_name_validation(self):
        """Test extracting validation job name."""
        result = extract_job_name_from_text("run validation")
        assert result == "data_validation"

    def test_extract_job_name_default(self):
        """Test extracting default job name."""
        result = extract_job_name_from_text("run unknown job")
        assert result == "data_processing"

    def test_extract_job_id_valid(self):
        """Test extracting valid job ID."""
        result = extract_job_id_from_text("debug job_003")
        assert result == "job_003"

    def test_extract_job_id_invalid(self):
        """Test extracting invalid job ID."""
        result = extract_job_id_from_text("debug job_abc")
        assert result is None

    def test_extract_job_id_none(self):
        """Test extracting job ID from text without job ID."""
        result = extract_job_id_from_text("debug something")
        assert result is None
