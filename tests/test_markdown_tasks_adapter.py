from __future__ import annotations

import os
from datetime import datetime, timezone

from graph.adapters.markdown_tasks import MarkdownTasksAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_markdown_tasks_ingests_checked_and_unchecked_tasks_with_metadata(tmp_path):
    note = tmp_path / "Projects.md"
    note.write_text(
        "# Project Alpha\n\n"
        "## Next actions\n"
        "- [ ] Draft import adapter due:2026-05-05\n"
        "- [x] Add registry entry @date(2026-05-06)\n"
        "Not a task\n",
        encoding="utf-8",
    )

    result = MarkdownTasksAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    first, second = result.units
    assert first.source_project == SourceProject.MARKDOWN_TASKS
    assert first.source_entity_type == "markdown_task"
    assert first.content_type == ContentType.ARTIFACT
    assert first.title == "Draft import adapter due:2026-05-05"
    assert first.content == first.title
    assert first.tags == ["markdown-task"]
    assert first.metadata == {
        "source_file": "Projects.md",
        "line_number": 4,
        "completed": False,
        "heading_path": ["Project Alpha", "Next actions"],
        "due": "2026-05-05",
    }
    assert second.metadata == {
        "source_file": "Projects.md",
        "line_number": 5,
        "completed": True,
        "heading_path": ["Project Alpha", "Next actions"],
        "date": "2026-05-06",
    }
    assert result.edges == []


def test_markdown_tasks_discovers_markdown_files_recursively_and_deterministically(tmp_path):
    root = tmp_path / "vault"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "b.md").write_text("- [ ] Root task\n", encoding="utf-8")
    (nested / "a.md").write_text("- [X] Nested task [due:: 2026-06-01]\n", encoding="utf-8")
    (nested / "ignored.txt").write_text("- [ ] Ignored\n", encoding="utf-8")

    result = MarkdownTasksAdapter(path=str(root)).ingest()

    assert [unit.metadata["source_file"] for unit in result.units] == [
        "b.md",
        "nested/a.md",
    ]
    assert [unit.title for unit in result.units] == [
        "Root task",
        "Nested task [due:: 2026-06-01]",
    ]
    assert result.units[1].metadata["completed"] is True
    assert result.units[1].metadata["due"] == "2026-06-01"
    assert len({unit.source_id for unit in result.units}) == 2
    assert all(unit.source_id.startswith("markdown_tasks:") for unit in result.units)


def test_markdown_tasks_preserves_heading_context_across_levels(tmp_path):
    note = tmp_path / "Plan.md"
    note.write_text(
        "# Area\n"
        "## Project\n"
        "- [ ] Under project\n"
        "### Detail\n"
        "- [ ] Under detail\n"
        "## Other project\n"
        "- [ ] Under other project\n",
        encoding="utf-8",
    )

    result = MarkdownTasksAdapter(path=str(note)).ingest()

    assert [unit.metadata["heading_path"] for unit in result.units] == [
        ["Area", "Project"],
        ["Area", "Project", "Detail"],
        ["Area", "Other project"],
    ]


def test_markdown_tasks_ignores_malformed_items_and_fenced_examples(tmp_path):
    note = tmp_path / "Mixed.md"
    note.write_text(
        "- [] Missing space\n"
        "- [y] Unsupported state\n"
        "```\n"
        "- [ ] Example only\n"
        "```\n"
        "* [ ] Valid bullet\n"
        "+ [x] Valid plus\n",
        encoding="utf-8",
    )

    result = MarkdownTasksAdapter(path=str(note)).ingest()

    assert [unit.title for unit in result.units] == ["Valid bullet", "Valid plus"]
    assert [unit.metadata["line_number"] for unit in result.units] == [6, 7]


def test_markdown_tasks_filters_entity_type_and_since(tmp_path):
    old_note = tmp_path / "old.md"
    new_note = tmp_path / "new.md"
    old_note.write_text("- [ ] Old\n", encoding="utf-8")
    new_note.write_text("- [ ] New\n", encoding="utf-8")
    old_time = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    new_time = datetime(2025, 1, 3, tzinfo=timezone.utc).timestamp()
    os.utime(old_note, (old_time, old_time))
    os.utime(new_note, (new_time, new_time))

    skipped = MarkdownTasksAdapter(path=str(tmp_path)).ingest(
        entity_types=["markdown_link"]
    )
    assert skipped.units == []
    assert skipped.edges == []

    result = MarkdownTasksAdapter(path=str(tmp_path)).ingest(
        since=SyncState(
            source_project="markdown_tasks",
            source_entity_type="markdown_task",
            last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
    )

    assert [unit.title for unit in result.units] == ["New"]


def test_markdown_tasks_missing_path_returns_empty_result(tmp_path):
    result = MarkdownTasksAdapter(path=str(tmp_path / "missing")).ingest()

    assert result.units == []
    assert result.edges == []


def test_markdown_tasks_adapter_is_registered():
    assert "markdown_tasks" in list_adapters()
    adapter = get_adapter("markdown_tasks", path="/tmp/notes")
    assert isinstance(adapter, MarkdownTasksAdapter)
    assert adapter.name == "markdown_tasks"
