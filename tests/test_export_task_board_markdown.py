from __future__ import annotations

from datetime import datetime, timedelta, timezone

from graph.export import export_task_board_markdown
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

BASE_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    title: str,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    source_project: SourceProject | str = SourceProject.GOOGLE_TASKS,
    content: str = "Task content",
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="task",
        title=title,
        content=content,
        content_type=ContentType.ARTIFACT,
        metadata=metadata or {},
        tags=tags or [],
        created_at=BASE_TIME,
        ingested_at=BASE_TIME,
        updated_at=BASE_TIME,
    )


def test_task_board_groups_by_completion_and_due_date():
    today = datetime.now(timezone.utc).date()
    report = export_task_board_markdown(
        [
            unit(
                "overdue",
                title="Overdue",
                metadata={"due_date": (today - timedelta(days=1)).isoformat()},
            ),
            unit("today", title="Today", metadata={"due_date": today.isoformat()}),
            unit(
                "future",
                title="Future",
                metadata={"due_date": (today + timedelta(days=1)).isoformat()},
            ),
            unit("none", title="No Due", metadata={"status": "open"}),
            unit(
                "done",
                title="Done",
                metadata={
                    "status": "completed",
                    "due_date": (today - timedelta(days=3)).isoformat(),
                },
            ),
        ]
    )

    assert report.index("## Overdue") < report.index("**Overdue**")
    assert report.index("## Due Today") < report.index("**Today**")
    assert report.index("## Upcoming") < report.index("**Future**")
    assert report.index("## No Due Date") < report.index("**No Due**")
    assert report.index("## Completed") < report.index("**Done**")


def test_task_board_detects_completed_boolean_and_invalid_dates():
    report = export_task_board_markdown(
        [
            unit("done", title="Boolean Done", metadata={"completed": True, "due": "not-a-date"}),
            unit("bad", title="Bad Due", metadata={"due": "not-a-date"}),
        ]
    )

    assert "## Completed" in report
    assert "**Boolean Done** - source: google_tasks; due: not-a-date" in report
    assert "## No Due Date" in report
    assert "**Bad Due** - source: google_tasks; due: not-a-date" in report


def test_task_board_output_is_deterministic_by_due_date_then_title():
    today = datetime.now(timezone.utc).date()
    later = (today + timedelta(days=3)).isoformat()
    sooner = (today + timedelta(days=2)).isoformat()
    tasks = [
        unit("b", title="Beta", metadata={"due_date": later}),
        unit("a", title="Alpha", metadata={"due_date": sooner}),
        unit("c", title="Aardvark", metadata={"due_date": later}),
    ]

    first = export_task_board_markdown(tasks)
    second = export_task_board_markdown(reversed(tasks))

    assert first == second
    assert first.index("**Alpha**") < first.index("**Aardvark**") < first.index("**Beta**")


def test_task_board_writes_to_file(tmp_path):
    path = tmp_path / "task-board.md"

    stats = export_task_board_markdown(
        [unit("a", title="Write Me", metadata={"status": "open"})],
        path,
    )

    assert stats == {
        "path": str(path),
        "tasks_exported": 1,
        "bytes_written": path.stat().st_size,
    }
    assert "**Write Me**" in path.read_text(encoding="utf-8")


def test_task_board_is_importable_from_graph_export():
    from graph.export import export_task_board_markdown as imported

    assert imported is export_task_board_markdown
