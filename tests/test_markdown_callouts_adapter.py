from __future__ import annotations

import os
from datetime import datetime, timezone

from graph.adapters.markdown_callouts import MarkdownCalloutsAdapter
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_markdown_callouts_extracts_multiple_callouts_in_source_order(tmp_path):
    note = tmp_path / "Research.md"
    note.write_text(
        "# Research\n\n"
        "> [!note] First signal\n"
        "> Alpha insight.\n"
        "> Continued detail.\n\n"
        "Between blocks.\n\n"
        "> [!warning] Second signal\n"
        "> Beta risk.\n",
        encoding="utf-8",
    )

    result = MarkdownCalloutsAdapter(path=str(note)).ingest()

    assert len(result.units) == 2
    first, second = result.units
    assert [unit.title for unit in result.units] == ["First signal", "Second signal"]
    assert [unit.metadata["line_number"] for unit in result.units] == [3, 9]
    assert first.source_project == SourceProject.MARKDOWN_CALLOUTS
    assert first.source_entity_type == "markdown_callout"
    assert first.content_type == ContentType.INSIGHT
    assert first.content == "Alpha insight.\nContinued detail."
    assert first.metadata["callout_type"] == "note"
    assert first.metadata["title"] == "First signal"
    assert first.metadata["body"] == first.content
    assert first.metadata["source_path"] == "Research.md"
    assert first.metadata["path"] == "Research.md"
    assert second.metadata["callout_type"] == "warning"
    assert result.edges == []


def test_markdown_callouts_preserves_nested_quoted_lines(tmp_path):
    note = tmp_path / "Nested.md"
    note.write_text(
        "> [!quote] Evidence\n"
        "> Outer quote.\n"
        "> > Nested quote stays quoted.\n"
        "> > [!tip] Nested marker is body text.\n"
        "> Back outside.\n",
        encoding="utf-8",
    )

    result = MarkdownCalloutsAdapter(path=str(note)).ingest()

    assert len(result.units) == 1
    assert result.units[0].content == (
        "Outer quote.\n"
        "> Nested quote stays quoted.\n"
        "> [!tip] Nested marker is body text.\n"
        "Back outside."
    )


def test_markdown_callouts_handles_missing_titles(tmp_path):
    note = tmp_path / "Untitled.md"
    note.write_text(
        "> [!idea]\n"
        "> Use callouts as focused retrieval units.\n",
        encoding="utf-8",
    )

    result = MarkdownCalloutsAdapter(path=str(note)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Idea callout"
    assert unit.metadata["title"] == ""
    assert unit.metadata["callout_type"] == "idea"


def test_markdown_callouts_records_heading_context(tmp_path):
    root = tmp_path / "vault"
    nested = root / "Notes"
    nested.mkdir(parents=True)
    note = nested / "Context.md"
    note.write_text(
        "# Project Alpha\n\n"
        "## Findings\n\n"
        "> [!success]+ Clear win\n"
        "> Retrieval quality improved.\n\n"
        "### Follow-up\n\n"
        "> [!todo]- Validate\n"
        "> Run on the full vault.\n",
        encoding="utf-8",
    )

    result = MarkdownCalloutsAdapter(
        root_path=str(nested),
        source_id_root=str(root),
    ).ingest()

    assert [unit.metadata["source_path"] for unit in result.units] == [
        "Notes/Context.md",
        "Notes/Context.md",
    ]
    assert result.units[0].metadata["heading"] == "Findings"
    assert result.units[0].metadata["headings"] == ["Project Alpha", "Findings"]
    assert result.units[0].metadata["modifier"] == "+"
    assert result.units[1].metadata["heading"] == "Follow-up"
    assert result.units[1].metadata["headings"] == [
        "Project Alpha",
        "Findings",
        "Follow-up",
    ]
    assert result.units[1].metadata["modifier"] == "-"


def test_markdown_callouts_filters_entity_type_and_since(tmp_path):
    old_note = tmp_path / "Old.md"
    old_note.write_text("> [!note] Old\n> Skip.\n", encoding="utf-8")
    new_note = tmp_path / "New.md"
    new_note.write_text("> [!note] New\n> Include.\n", encoding="utf-8")
    old_time = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    new_time = datetime(2025, 1, 3, tzinfo=timezone.utc).timestamp()
    old_note.touch()
    new_note.touch()
    os.utime(old_note, (old_time, old_time))
    os.utime(new_note, (new_time, new_time))

    skipped = MarkdownCalloutsAdapter(path=str(tmp_path)).ingest(
        entity_types=["markdown_link"]
    )
    assert skipped.units == []
    assert skipped.edges == []

    result = MarkdownCalloutsAdapter(path=str(tmp_path)).ingest(
        since=SyncState(
            source_project="markdown_callouts",
            source_entity_type="markdown_callout",
            last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
    )

    assert [unit.title for unit in result.units] == ["New"]


def test_markdown_callouts_missing_path_returns_empty_result(tmp_path):
    result = MarkdownCalloutsAdapter(path=str(tmp_path / "missing")).ingest()

    assert result.units == []
    assert result.edges == []
