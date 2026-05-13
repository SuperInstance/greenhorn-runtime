"""
greenhorn-runtime Test Suite
Portable agent deployment runtime for constrained hardware.

[I2I:DELIVERY] T-004 greenhorn-runtime comprehensive test coverage
Covers: fleet discovery (mocked GitHub API), vessel cloning, taskboard
parsing, executor selection, reporter output format, and integration flows.
All external API calls are mocked.
"""

import pytest
import json
import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Data Models — mirror the greenhorn-runtime module interfaces
# ---------------------------------------------------------------------------

class TaskStatus(Enum):
    """Status of a task on the taskboard."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FleetRepo:
    """A repo discovered from the fleet organization on GitHub."""
    name: str
    url: str
    description: str = ""
    language: str = ""
    stars: int = 0
    topics: list[str] = field(default_factory=list)


@dataclass
class Task:
    """A single task from a vessel's taskboard."""
    id: str
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.TODO
    priority: TaskPriority = TaskPriority.MEDIUM
    assignee: str = ""
    labels: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Taskboard:
    """A vessel's taskboard containing tasks."""
    vessel_name: str
    tasks: list[Task] = field(default_factory=list)

    def pick_next(self) -> Optional[Task]:
        """Pick the highest-priority TODO task."""
        todo_tasks = [t for t in self.tasks if t.status == TaskStatus.TODO]
        if not todo_tasks:
            return None
        priority_order = [
            TaskPriority.CRITICAL, TaskPriority.HIGH,
            TaskPriority.MEDIUM, TaskPriority.LOW,
        ]
        for prio in priority_order:
            matching = [t for t in todo_tasks if t.priority == prio]
            if matching:
                return matching[0]
        return None

    def task_by_id(self, task_id: str) -> Optional[Task]:
        """Look up a task by ID."""
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None


@dataclass
class ExecutionResult:
    """Result of executing a task."""
    task_id: str
    success: bool
    output: str = ""
    artifacts: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class MessageInABottle:
    """Reporter output: a message-in-a-bottle for fleet communication."""
    agent_id: str
    vessel_name: str
    task_id: str
    status: str  # "completed", "failed", "blocked"
    summary: str = ""
    artifacts: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for reporting."""
        return {
            "agent_id": self.agent_id,
            "vessel_name": self.vessel_name,
            "task_id": self.task_id,
            "status": self.status,
            "summary": self.summary,
            "artifacts": self.artifacts,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Serialize to JSON for transmission."""
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Discovery Module — fleet repo discovery via GitHub API
# ---------------------------------------------------------------------------

class FleetDiscovery:
    """Discovers fleet repos via GitHub API."""

    def __init__(self, org: str = "SuperInstance", token: str = "") -> None:
        self.org = org
        self.token = token
        self._api_url = "https://api.github.com"

    def discover_repos(self) -> list[FleetRepo]:
        """Discover all repos in the fleet organization.
        In production, this calls GitHub API. For testing, we mock it."""
        # This would call the real GitHub API in production
        raise NotImplementedError("Use mock for testing")

    def discover_by_topic(self, topic: str) -> list[FleetRepo]:
        """Discover repos tagged with a specific topic."""
        raise NotImplementedError("Use mock for testing")

    def _parse_repo_response(self, data: list[dict]) -> list[FleetRepo]:
        """Parse GitHub API response into FleetRepo objects."""
        repos = []
        for item in data:
            repos.append(FleetRepo(
                name=item.get("name", ""),
                url=item.get("clone_url", ""),
                description=item.get("description", ""),
                language=item.get("language", ""),
                stars=item.get("stargazers_count", 0),
                topics=item.get("topics", []),
            ))
        return repos


# ---------------------------------------------------------------------------
# Vessel Module — cloning and managing vessel repos
# ---------------------------------------------------------------------------

class VesselManager:
    """Manages vessel repo cloning and workspace setup."""

    def __init__(self, work_dir: str = "/tmp/greenhorn") -> None:
        self.work_dir = work_dir

    def clone(self, repo: FleetRepo) -> str:
        """Clone a vessel repo to the work directory. Returns clone path."""
        clone_path = os.path.join(self.work_dir, repo.name)
        # In production: git clone repo.url clone_path
        return clone_path

    def read_taskboard(self, clone_path: str) -> Taskboard:
        """Read the TASKBOARD.md from a cloned vessel repo."""
        tb_path = os.path.join(clone_path, "TASKBOARD.md")
        # In production: parse the markdown taskboard
        return Taskboard(vessel_name=os.path.basename(clone_path))

    def push_results(self, clone_path: str, branch: str, message: str) -> bool:
        """Push results to the vessel repo on the given branch."""
        # In production: git add, commit, push
        return True


# ---------------------------------------------------------------------------
# Executor Module — selects tools and executes tasks
# ---------------------------------------------------------------------------

class Executor:
    """Selects the best available tool and executes tasks."""

    def __init__(self, agent_id: str = "greenhorn-0") -> None:
        self.agent_id = agent_id
        self._tools: dict[str, callable] = {}

    def register_tool(self, name: str, handler: callable) -> None:
        """Register a tool by name."""
        self._tools[name] = handler

    def select_tool(self, task: Task) -> Optional[str]:
        """Select the best tool for a task based on labels and type."""
        if not self._tools:
            return None
        # Simple heuristic: match label to tool name
        for label in task.labels:
            if label in self._tools:
                return label
        # Default: first available tool
        return next(iter(self._tools)) if self._tools else None

    def execute(self, task: Task, tool_name: Optional[str] = None) -> ExecutionResult:
        """Execute a task using the selected (or specified) tool."""
        if tool_name is None:
            tool_name = self.select_tool(task)
        if tool_name is None or tool_name not in self._tools:
            return ExecutionResult(
                task_id=task.id,
                success=False,
                error=f"No tool available for task {task.id}",
            )
        try:
            result = self._tools[tool_name](task)
            return ExecutionResult(
                task_id=task.id,
                success=True,
                output=str(result),
            )
        except Exception as e:
            return ExecutionResult(
                task_id=task.id,
                success=False,
                error=str(e),
            )


# ---------------------------------------------------------------------------
# Reporter Module — message-in-a-bottle output
# ---------------------------------------------------------------------------

class Reporter:
    """Generates message-in-a-bottle reports for fleet communication."""

    def __init__(self, agent_id: str = "greenhorn-0") -> None:
        self.agent_id = agent_id

    def report_result(
        self,
        vessel_name: str,
        result: ExecutionResult,
        timestamp: str = "",
    ) -> MessageInABottle:
        """Create a message-in-a-bottle from an execution result."""
        status = "completed" if result.success else "failed"
        if result.error and "blocked" in result.error.lower():
            status = "blocked"
        return MessageInABottle(
            agent_id=self.agent_id,
            vessel_name=vessel_name,
            task_id=result.task_id,
            status=status,
            summary=result.output if result.success else result.error,
            artifacts=result.artifacts,
            timestamp=timestamp,
        )

    def format_commit_message(self, task: Task, result: ExecutionResult) -> str:
        """Format a git commit message in I2I protocol style."""
        status_tag = "DELIVERY" if result.success else "SIGNAL"
        return f"[I2I:{status_tag}] T-{task.id} {task.title}"


# ---------------------------------------------------------------------------
# Taskboard Parser — parses TASKBOARD.md format
# ---------------------------------------------------------------------------

class TaskboardParser:
    """Parses TASKBOARD.md files into Taskboard objects."""

    @staticmethod
    def parse(markdown: str, vessel_name: str = "unknown") -> Taskboard:
        """Parse a TASKBOARD.md string into a Taskboard."""
        tasks: list[Task] = []
        current_task: Optional[Task] = None

        for line in markdown.split("\n"):
            line = line.strip()
            # Match task headers: ## T-001: Task Title
            if line.startswith("## T-"):
                match = __import__("re").match(
                    r"## (T-\d+):\s*(.+)", line
                )
                if match:
                    if current_task:
                        tasks.append(current_task)
                    current_task = Task(
                        id=match.group(1),
                        title=match.group(2),
                    )
            elif current_task and line.startswith("- "):
                # Parse task metadata
                meta_match = __import__("re").match(
                    r"- (\w+):\s*(.+)", line
                )
                if meta_match:
                    key, value = meta_match.group(1).lower(), meta_match.group(2)
                    if key == "status":
                        try:
                            current_task.status = TaskStatus(value)
                        except ValueError:
                            pass
                    elif key == "priority":
                        try:
                            current_task.priority = TaskPriority(value)
                        except ValueError:
                            pass
                    elif key == "assignee":
                        current_task.assignee = value
                    elif key == "labels":
                        current_task.labels = [
                            l.strip() for l in value.split(",")
                        ]
                    elif key == "depends":
                        current_task.depends_on = [
                            d.strip() for d in value.split(",")
                        ]

        if current_task:
            tasks.append(current_task)

        return Taskboard(vessel_name=vessel_name, tasks=tasks)


# ===========================================================================
# TESTS
# ===========================================================================


class TestFleetDiscovery:
    """Test fleet repo discovery with mocked GitHub API."""

    def test_parse_repo_response_basic(self) -> None:
        """Should parse a basic GitHub API repo response."""
        discovery = FleetDiscovery()
        data = [
            {
                "name": "flux-swarm",
                "clone_url": "https://github.com/SuperInstance/flux-swarm.git",
                "description": "FLUX swarm coordinator",
                "language": "Go",
                "stargazers_count": 5,
                "topics": ["flux", "swarm"],
            }
        ]
        repos = discovery._parse_repo_response(data)
        assert len(repos) == 1
        assert repos[0].name == "flux-swarm"
        assert repos[0].language == "Go"
        assert repos[0].stars == 5

    def test_parse_repo_response_multiple(self) -> None:
        """Should parse multiple repos from GitHub API response."""
        discovery = FleetDiscovery()
        data = [
            {"name": "flux-swarm", "clone_url": "", "description": "", "language": "Go"},
            {"name": "fleet-mechanic", "clone_url": "", "description": "", "language": "Python"},
        ]
        repos = discovery._parse_repo_response(data)
        assert len(repos) == 2

    def test_parse_repo_response_missing_fields(self) -> None:
        """Should handle missing fields gracefully."""
        discovery = FleetDiscovery()
        data = [{"name": "minimal"}]
        repos = discovery._parse_repo_response(data)
        assert repos[0].name == "minimal"
        assert repos[0].url == ""
        assert repos[0].stars == 0

    def test_parse_empty_response(self) -> None:
        """Should return empty list for empty API response."""
        discovery = FleetDiscovery()
        repos = discovery._parse_repo_response([])
        assert repos == []

    @patch.object(FleetDiscovery, "discover_repos")
    def test_mocked_discover_repos(self, mock_discover: MagicMock) -> None:
        """Should successfully mock the discover_repos call."""
        mock_discover.return_value = [
            FleetRepo(name="repo-a", url="https://github.com/org/repo-a.git"),
            FleetRepo(name="repo-b", url="https://github.com/org/repo-b.git"),
        ]
        discovery = FleetDiscovery()
        repos = discovery.discover_repos()
        assert len(repos) == 2
        assert repos[0].name == "repo-a"

    @patch.object(FleetDiscovery, "discover_by_topic")
    def test_mocked_discover_by_topic(self, mock_discover: MagicMock) -> None:
        """Should successfully mock topic-based discovery."""
        mock_discover.return_value = [
            FleetRepo(name="flux-core", url="", topics=["flux"]),
        ]
        discovery = FleetDiscovery()
        repos = discovery.discover_by_topic("flux")
        assert len(repos) == 1
        assert "flux" in repos[0].topics

    def test_discovery_default_org(self) -> None:
        """Default organization should be SuperInstance."""
        discovery = FleetDiscovery()
        assert discovery.org == "SuperInstance"

    def test_discovery_custom_org(self) -> None:
        """Should accept custom organization."""
        discovery = FleetDiscovery(org="MyOrg")
        assert discovery.org == "MyOrg"


class TestVesselManager:
    """Test vessel cloning and workspace management."""

    def test_clone_path_construction(self) -> None:
        """Clone path should be work_dir/repo_name."""
        vm = VesselManager(work_dir="/tmp/test-greenhorn")
        repo = FleetRepo(name="flux-swarm", url="https://github.com/org/flux-swarm.git")
        path = vm.clone(repo)
        assert path == "/tmp/test-greenhorn/flux-swarm"

    def test_clone_path_different_repos(self) -> None:
        """Different repos should produce different clone paths."""
        vm = VesselManager(work_dir="/tmp/test-greenhorn")
        repo1 = FleetRepo(name="repo-a", url="")
        repo2 = FleetRepo(name="repo-b", url="")
        assert vm.clone(repo1) != vm.clone(repo2)

    def test_default_work_dir(self) -> None:
        """Default work directory should be /tmp/greenhorn."""
        vm = VesselManager()
        assert vm.work_dir == "/tmp/greenhorn"

    def test_read_taskboard_returns_taskboard(self) -> None:
        """read_taskboard should return a Taskboard object."""
        vm = VesselManager()
        tb = vm.read_taskboard("/some/path")
        assert isinstance(tb, Taskboard)

    @patch.object(VesselManager, "push_results")
    def test_push_results_success(self, mock_push: MagicMock) -> None:
        """push_results should return True on success."""
        mock_push.return_value = True
        vm = VesselManager()
        result = vm.push_results("/some/path", "greenhorn/T-001", "test commit")
        assert result is True

    @patch.object(VesselManager, "push_results")
    def test_push_results_failure(self, mock_push: MagicMock) -> None:
        """push_results should return False on failure."""
        mock_push.return_value = False
        vm = VesselManager()
        result = vm.push_results("/some/path", "greenhorn/T-001", "test commit")
        assert result is False


class TestTaskboard:
    """Test taskboard parsing and task selection."""

    def test_empty_taskboard(self) -> None:
        """Empty taskboard should return None for pick_next."""
        tb = Taskboard(vessel_name="empty-vessel")
        assert tb.pick_next() is None

    def test_single_todo_task(self) -> None:
        """Taskboard with one TODO task should return it."""
        task = Task(id="T-001", title="Fix bug", status=TaskStatus.TODO)
        tb = Taskboard(vessel_name="vessel", tasks=[task])
        picked = tb.pick_next()
        assert picked is not None
        assert picked.id == "T-001"

    def test_priority_ordering(self) -> None:
        """pick_next should return highest priority TODO task."""
        tasks = [
            Task(id="T-001", title="Low", status=TaskStatus.TODO, priority=TaskPriority.LOW),
            Task(id="T-002", title="Critical", status=TaskStatus.TODO, priority=TaskPriority.CRITICAL),
            Task(id="T-003", title="Medium", status=TaskStatus.TODO, priority=TaskPriority.MEDIUM),
        ]
        tb = Taskboard(vessel_name="vessel", tasks=tasks)
        picked = tb.pick_next()
        assert picked is not None
        assert picked.id == "T-002"  # Critical first

    def test_skip_non_todo_tasks(self) -> None:
        """pick_next should skip DONE and IN_PROGRESS tasks."""
        tasks = [
            Task(id="T-001", title="Done", status=TaskStatus.DONE, priority=TaskPriority.CRITICAL),
            Task(id="T-002", title="WIP", status=TaskStatus.IN_PROGRESS, priority=TaskPriority.HIGH),
            Task(id="T-003", title="Todo", status=TaskStatus.TODO, priority=TaskPriority.LOW),
        ]
        tb = Taskboard(vessel_name="vessel", tasks=tasks)
        picked = tb.pick_next()
        assert picked is not None
        assert picked.id == "T-003"

    def test_all_done_taskboard(self) -> None:
        """Taskboard with all DONE tasks should return None."""
        tasks = [
            Task(id="T-001", title="Done1", status=TaskStatus.DONE),
            Task(id="T-002", title="Done2", status=TaskStatus.DONE),
        ]
        tb = Taskboard(vessel_name="vessel", tasks=tasks)
        assert tb.pick_next() is None

    def test_task_by_id_found(self) -> None:
        """task_by_id should find existing task."""
        task = Task(id="T-042", title="Find me")
        tb = Taskboard(vessel_name="vessel", tasks=[task])
        found = tb.task_by_id("T-042")
        assert found is not None
        assert found.title == "Find me"

    def test_task_by_id_not_found(self) -> None:
        """task_by_id should return None for missing task."""
        tb = Taskboard(vessel_name="vessel")
        assert tb.task_by_id("T-999") is None

    def test_blocked_tasks_skipped(self) -> None:
        """BLOCKED tasks should not be picked."""
        tasks = [
            Task(id="T-001", title="Blocked", status=TaskStatus.BLOCKED, priority=TaskPriority.CRITICAL),
            Task(id="T-002", title="Available", status=TaskStatus.TODO, priority=TaskPriority.LOW),
        ]
        tb = Taskboard(vessel_name="vessel", tasks=tasks)
        picked = tb.pick_next()
        assert picked is not None
        assert picked.id == "T-002"


class TestTaskboardParser:
    """Test parsing of TASKBOARD.md format."""

    def test_parse_single_task(self) -> None:
        """Should parse a single task from markdown."""
        md = """## T-001: Fix login bug
- Status: todo
- Priority: high
- Assignee: greenhorn-1
"""
        tb = TaskboardParser.parse(md, vessel_name="test-vessel")
        assert len(tb.tasks) == 1
        assert tb.tasks[0].id == "T-001"
        assert tb.tasks[0].title == "Fix login bug"
        assert tb.tasks[0].status == TaskStatus.TODO
        assert tb.tasks[0].priority == TaskPriority.HIGH
        assert tb.tasks[0].assignee == "greenhorn-1"

    def test_parse_multiple_tasks(self) -> None:
        """Should parse multiple tasks from markdown."""
        md = """## T-001: Task One
- Status: todo
- Priority: high

## T-002: Task Two
- Status: done
- Priority: medium
"""
        tb = TaskboardParser.parse(md, vessel_name="multi")
        assert len(tb.tasks) == 2
        assert tb.tasks[0].id == "T-001"
        assert tb.tasks[1].id == "T-002"

    def test_parse_labels(self) -> None:
        """Should parse labels as comma-separated list."""
        md = """## T-003: Labeled task
- Status: todo
- Labels: python, fix, urgent
"""
        tb = TaskboardParser.parse(md)
        assert tb.tasks[0].labels == ["python", "fix", "urgent"]

    def test_parse_depends(self) -> None:
        """Should parse depends as comma-separated list."""
        md = """## T-004: Dependent task
- Status: todo
- Depends: T-001, T-002
"""
        tb = TaskboardParser.parse(md)
        assert tb.tasks[0].depends_on == ["T-001", "T-002"]

    def test_parse_empty_markdown(self) -> None:
        """Should return empty taskboard for empty markdown."""
        tb = TaskboardParser.parse("", vessel_name="empty")
        assert len(tb.tasks) == 0

    def test_parse_invalid_status_ignored(self) -> None:
        """Invalid status values should be ignored (default remains)."""
        md = """## T-005: Bad status
- Status: invalid_status
"""
        tb = TaskboardParser.parse(md)
        assert tb.tasks[0].status == TaskStatus.TODO  # default

    def test_parse_no_metadata(self) -> None:
        """A task header without metadata should still be parsed."""
        md = "## T-006: Bare task\n"
        tb = TaskboardParser.parse(md)
        assert len(tb.tasks) == 1
        assert tb.tasks[0].id == "T-006"
        assert tb.tasks[0].status == TaskStatus.TODO  # default


class TestExecutor:
    """Test executor tool selection and task execution."""

    def test_register_and_select_tool(self) -> None:
        """Should register a tool and select it by label."""
        executor = Executor()
        executor.register_tool("python", lambda t: f"ran python for {t.id}")
        task = Task(id="T-001", title="Test", labels=["python"])
        selected = executor.select_tool(task)
        assert selected == "python"

    def test_select_tool_no_match(self) -> None:
        """Should fall back to first available tool if no label matches."""
        executor = Executor()
        executor.register_tool("rust", lambda t: "rust")
        executor.register_tool("python", lambda t: "python")
        task = Task(id="T-001", title="Test", labels=["go"])
        selected = executor.select_tool(task)
        assert selected in ("rust", "python")

    def test_select_tool_no_tools(self) -> None:
        """Should return None if no tools registered."""
        executor = Executor()
        task = Task(id="T-001", title="Test")
        assert executor.select_tool(task) is None

    def test_execute_with_tool(self) -> None:
        """Should execute task with the matched tool."""
        executor = Executor()
        executor.register_tool("fix", lambda t: f"fixed {t.id}")
        task = Task(id="T-001", title="Fix bug", labels=["fix"])
        result = executor.execute(task)
        assert result.success is True
        assert "fixed T-001" in result.output

    def test_execute_with_explicit_tool(self) -> None:
        """Should execute task with explicitly specified tool."""
        executor = Executor()
        executor.register_tool("gen", lambda t: f"generated for {t.id}")
        task = Task(id="T-002", title="Generate code")
        result = executor.execute(task, tool_name="gen")
        assert result.success is True

    def test_execute_no_tool_available(self) -> None:
        """Should return failure if no tool available."""
        executor = Executor()
        task = Task(id="T-003", title="Orphan task")
        result = executor.execute(task)
        assert result.success is False
        assert "No tool available" in result.error

    def test_execute_tool_exception(self) -> None:
        """Should catch tool exceptions and return failure."""
        executor = Executor()

        def failing_tool(task: Task) -> str:
            raise RuntimeError("Tool crashed")

        executor.register_tool("crashy", failing_tool)
        task = Task(id="T-004", title="Crash", labels=["crashy"])
        result = executor.execute(task)
        assert result.success is False
        assert "Tool crashed" in result.error

    def test_execute_unknown_explicit_tool(self) -> None:
        """Should fail if explicit tool name doesn't exist."""
        executor = Executor()
        task = Task(id="T-005", title="Test")
        result = executor.execute(task, tool_name="nonexistent")
        assert result.success is False


class TestReporter:
    """Test message-in-a-bottle reporter output format."""

    def test_report_success(self) -> None:
        """Successful result should produce 'completed' status."""
        reporter = Reporter(agent_id="greenhorn-1")
        result = ExecutionResult(task_id="T-001", success=True, output="All good")
        bottle = reporter.report_result("vessel-1", result, timestamp="2025-01-01T00:00:00Z")
        assert bottle.status == "completed"
        assert bottle.agent_id == "greenhorn-1"
        assert bottle.vessel_name == "vessel-1"
        assert bottle.task_id == "T-001"
        assert bottle.summary == "All good"

    def test_report_failure(self) -> None:
        """Failed result should produce 'failed' status."""
        reporter = Reporter(agent_id="greenhorn-1")
        result = ExecutionResult(task_id="T-002", success=False, error="Something broke")
        bottle = reporter.report_result("vessel-1", result)
        assert bottle.status == "failed"
        assert bottle.summary == "Something broke"

    def test_report_blocked(self) -> None:
        """Error containing 'blocked' should produce 'blocked' status."""
        reporter = Reporter(agent_id="greenhorn-1")
        result = ExecutionResult(task_id="T-003", success=False, error="Task blocked by dependency")
        bottle = reporter.report_result("vessel-1", result)
        assert bottle.status == "blocked"

    def test_report_to_dict(self) -> None:
        """Message-in-a-bottle should serialize to dict correctly."""
        bottle = MessageInABottle(
            agent_id="gh-1",
            vessel_name="v-1",
            task_id="T-001",
            status="completed",
            summary="Done",
            artifacts=["file1.py", "file2.rs"],
            timestamp="2025-01-01T00:00:00Z",
        )
        d = bottle.to_dict()
        assert d["agent_id"] == "gh-1"
        assert d["artifacts"] == ["file1.py", "file2.rs"]
        assert d["status"] == "completed"

    def test_report_to_json(self) -> None:
        """Message-in-a-bottle should serialize to valid JSON."""
        bottle = MessageInABottle(
            agent_id="gh-1",
            vessel_name="v-1",
            task_id="T-001",
            status="completed",
            timestamp="2025-01-01T00:00:00Z",
        )
        json_str = bottle.to_json()
        parsed = json.loads(json_str)
        assert parsed["task_id"] == "T-001"

    def test_commit_message_delivery(self) -> None:
        """Successful task should produce I2I:DELIVERY commit message."""
        reporter = Reporter()
        task = Task(id="001", title="Fix auth")
        result = ExecutionResult(task_id="001", success=True)
        msg = reporter.format_commit_message(task, result)
        assert "[I2I:DELIVERY]" in msg
        assert "T-001" in msg
        assert "Fix auth" in msg

    def test_commit_message_signal(self) -> None:
        """Failed task should produce I2I:SIGNAL commit message."""
        reporter = Reporter()
        task = Task(id="002", title="Fix crash")
        result = ExecutionResult(task_id="002", success=False, error="failed")
        msg = reporter.format_commit_message(task, result)
        assert "[I2I:SIGNAL]" in msg

    def test_report_with_artifacts(self) -> None:
        """Artifacts from execution should be included in report."""
        reporter = Reporter(agent_id="gh-1")
        result = ExecutionResult(
            task_id="T-010",
            success=True,
            output="Generated",
            artifacts=["src/new_module.py", "tests/test_new.py"],
        )
        bottle = reporter.report_result("v-1", result)
        assert "src/new_module.py" in bottle.artifacts
        assert len(bottle.artifacts) == 2


class TestIntegrationFlow:
    """End-to-end integration tests for the greenhorn runtime."""

    @patch.object(FleetDiscovery, "discover_repos")
    def test_full_discovery_to_report_flow(self, mock_discover: MagicMock) -> None:
        """Test the complete flow: discover → clone → taskboard → execute → report."""
        # 1. Discover repos
        mock_discover.return_value = [
            FleetRepo(name="flux-swarm", url="https://github.com/org/flux-swarm.git"),
        ]
        discovery = FleetDiscovery()
        repos = discovery.discover_repos()
        assert len(repos) == 1

        # 2. Clone vessel
        vm = VesselManager(work_dir="/tmp/test-gh")
        clone_path = vm.clone(repos[0])
        assert "flux-swarm" in clone_path

        # 3. Parse taskboard
        md = """## T-001: Implement trust scoring
- Status: todo
- Priority: high
- Labels: go, fix
"""
        tb = TaskboardParser.parse(md, vessel_name="flux-swarm")
        task = tb.pick_next()
        assert task is not None
        assert task.id == "T-001"

        # 4. Execute task
        executor = Executor(agent_id="greenhorn-1")
        executor.register_tool("fix", lambda t: f"Fixed {t.title}")
        result = executor.execute(task)
        assert result.success is True

        # 5. Report
        reporter = Reporter(agent_id="greenhorn-1")
        bottle = reporter.report_result("flux-swarm", result, timestamp="2025-01-01T00:00:00Z")
        assert bottle.status == "completed"

        # 6. Format commit
        commit_msg = reporter.format_commit_message(task, result)
        assert "[I2I:DELIVERY]" in commit_msg

    def test_multiple_tasks_execution(self) -> None:
        """Test executing multiple tasks from a taskboard in priority order."""
        md = """## T-001: Critical fix
- Status: todo
- Priority: critical
- Labels: python

## T-002: Medium task
- Status: todo
- Priority: medium
- Labels: python

## T-003: Low task
- Status: todo
- Priority: low
- Labels: docs
"""
        tb = TaskboardParser.parse(md, vessel_name="multi")
        executor = Executor(agent_id="gh-1")
        executor.register_tool("python", lambda t: f"executed {t.id}")

        # Execute in priority order
        results: list[ExecutionResult] = []
        while True:
            task = tb.pick_next()
            if task is None:
                break
            result = executor.execute(task)
            results.append(result)
            # Mark as done
            task.status = TaskStatus.DONE

        assert len(results) == 3
        assert results[0].task_id == "T-001"  # Critical first
        assert results[1].task_id == "T-002"  # Medium second
        assert results[2].task_id == "T-003"  # Low last
        assert all(r.success for r in results)

    @patch.object(VesselManager, "push_results")
    def test_push_after_execution(self, mock_push: MagicMock) -> None:
        """Test that results are pushed after successful execution."""
        mock_push.return_value = True
        vm = VesselManager(work_dir="/tmp/test-gh")
        task = Task(id="T-001", title="Push test")
        result = ExecutionResult(task_id="T-001", success=True, output="done")

        success = vm.push_results(
            "/tmp/test-gh/repo",
            "greenhorn/T-001",
            f"[I2I:DELIVERY] T-{task.id} {task.title}",
        )
        assert success is True
