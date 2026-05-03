from __future__ import annotations

import os
from datetime import datetime, timezone

from graph.adapters.markdown_definitions import MarkdownDefinitionsAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_markdown_definitions_ingests_definition_list_and_inline_patterns(tmp_path):
    note = tmp_path / "Glossary.md"
    note.write_text(
        "# Concepts\n\n"
        "Algorithm\n"
        ": A repeatable procedure for solving a problem #CS.\n\n"
        "Knowledge graph:: A network of typed entities #Graph/Model\n",
        encoding="utf-8",
    )

    result = MarkdownDefinitionsAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    first, second = result.units
    assert first.source_project == SourceProject.MARKDOWN_DEFINITIONS
    assert first.source_entity_type == "markdown_definition"
    assert first.content_type == ContentType.INSIGHT
    assert first.title == "Algorithm"
    assert first.content == "A repeatable procedure for solving a problem #CS."
    assert first.metadata == {
        "term": "Algorithm",
        "source_file": "Glossary.md",
        "file_path": "Glossary.md",
        "line_number": 3,
        "heading_path": ["Concepts"],
    }
    assert first.tags == ["cs"]
    assert second.title == "Knowledge graph"
    assert second.content == "A network of typed entities #Graph/Model"
    assert second.metadata["line_number"] == 6
    assert second.metadata["heading_path"] == ["Concepts"]
    assert second.tags == ["graph/model"]
    assert result.edges == []


def test_markdown_definitions_ingests_headings_followed_by_definition_paragraphs(
    tmp_path,
):
    note = tmp_path / "terms.md"
    note.write_text(
        "# Glossary\n\n"
        "## Latent space\n\n"
        "A compressed representational space used by a model. #ML\n"
        "It can group semantically similar inputs.\n\n"
        "## Empty term\n\n"
        "- Not a definition paragraph\n",
        encoding="utf-8",
    )

    result = MarkdownDefinitionsAdapter(path=str(note)).ingest()

    assert [unit.title for unit in result.units] == ["Latent space"]
    unit = result.units[0]
    assert unit.content == (
        "A compressed representational space used by a model. #ML "
        "It can group semantically similar inputs."
    )
    assert unit.metadata == {
        "term": "Latent space",
        "source_file": "terms.md",
        "file_path": "terms.md",
        "line_number": 3,
        "heading_path": ["Glossary", "Latent space"],
    }
    assert unit.tags == ["ml"]


def test_markdown_definitions_discovers_markdown_files_recursively(tmp_path):
    root = tmp_path / "vault"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / "b.md").write_text("Beta:: Second letter\n", encoding="utf-8")
    (nested / "a.markdown").write_text("Alpha:: First letter\n", encoding="utf-8")
    (nested / "ignored.txt").write_text("Gamma:: Ignored\n", encoding="utf-8")

    result = MarkdownDefinitionsAdapter(path=str(root)).ingest()

    assert [unit.metadata["source_file"] for unit in result.units] == [
        "b.md",
        "nested/a.markdown",
    ]
    assert [unit.title for unit in result.units] == ["Beta", "Alpha"]
    assert all(unit.source_id.startswith("markdown_definitions:") for unit in result.units)
    assert len({unit.source_id for unit in result.units}) == 2


def test_markdown_definitions_ignores_fenced_examples_and_bad_patterns(tmp_path):
    note = tmp_path / "mixed.md"
    note.write_text(
        "```\n"
        "Example:: Not ingested\n"
        "```\n"
        "URL: https://example.com\n"
        "Trailing colon:\n"
        ": Not a valid term\n"
        "Valid term:: Valid definition #Tag.\n",
        encoding="utf-8",
    )

    result = MarkdownDefinitionsAdapter(path=str(note)).ingest()

    assert [unit.title for unit in result.units] == ["Valid term"]
    assert result.units[0].tags == ["tag"]


def test_markdown_definitions_filters_entity_type_and_since(tmp_path):
    old_note = tmp_path / "old.md"
    new_note = tmp_path / "new.md"
    old_note.write_text("Old:: Definition\n", encoding="utf-8")
    new_note.write_text("New:: Definition\n", encoding="utf-8")
    old_time = datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp()
    new_time = datetime(2025, 1, 3, tzinfo=timezone.utc).timestamp()
    os.utime(old_note, (old_time, old_time))
    os.utime(new_note, (new_time, new_time))

    skipped = MarkdownDefinitionsAdapter(path=str(tmp_path)).ingest(
        entity_types=["markdown_task"]
    )
    assert skipped.units == []
    assert skipped.edges == []

    result = MarkdownDefinitionsAdapter(path=str(tmp_path)).ingest(
        since=SyncState(
            source_project="markdown_definitions",
            source_entity_type="markdown_definition",
            last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
    )

    assert [unit.title for unit in result.units] == ["New"]


def test_markdown_definitions_missing_path_returns_empty_result(tmp_path):
    result = MarkdownDefinitionsAdapter(path=str(tmp_path / "missing")).ingest()

    assert result.units == []
    assert result.edges == []


def test_markdown_definitions_adapter_is_registered():
    assert "markdown_definitions" in list_adapters()
    adapter = get_adapter("markdown_definitions", path="/tmp/notes")
    assert isinstance(adapter, MarkdownDefinitionsAdapter)
    assert adapter.name == "markdown_definitions"
