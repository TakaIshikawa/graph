from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.markdown_notes import MarkdownNotesAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject


def test_markdown_notes_ingests_valid_frontmatter_and_metadata(tmp_path):
    root = tmp_path / "vault"
    note_dir = root / "areas"
    note_dir.mkdir(parents=True)
    note = note_dir / "strategy.md"
    note.write_text(
        "---\n"
        "title: Strategy Note\n"
        "tags:\n"
        "  - Product\n"
        "  - '#Research'\n"
        "aliases: [Plan A, Alternate Strategy]\n"
        "created: 2026-01-02\n"
        "updated: 2026-01-03T04:05:06Z\n"
        "status: draft\n"
        "---\n"
        "# Ignored Heading\n\n"
        "This references [[Roadmap|the roadmap]] and [[People#Alice]].\n",
        encoding="utf-8",
    )

    result = MarkdownNotesAdapter(path=str(root)).ingest()

    assert len(result.units) == 1
    assert result.edges == []
    unit = result.units[0]
    assert unit.source_project == SourceProject.MARKDOWN_NOTES
    assert unit.source_entity_type == "markdown_note"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.title == "Strategy Note"
    assert unit.content.startswith("# Ignored Heading")
    assert unit.tags == ["product", "research"]
    assert unit.created_at == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 1, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert unit.metadata["source_file"] == "areas/strategy.md"
    assert unit.metadata["frontmatter"]["status"] == "draft"
    assert unit.metadata["aliases"] == ["Plan A", "Alternate Strategy"]
    assert unit.metadata["wikilinks"] == [
        {"raw": "Roadmap|the roadmap", "target": "Roadmap", "alias": "the roadmap"},
        {"raw": "People#Alice", "target": "People", "heading": "Alice"},
    ]


def test_markdown_notes_without_frontmatter_uses_heading_then_filename(tmp_path):
    heading_note = tmp_path / "has-heading.md"
    plain_note = tmp_path / "plain-note.md"
    heading_note.write_text("# Heading Title\n\nBody text\n", encoding="utf-8")
    plain_note.write_text("Body without a heading\n", encoding="utf-8")

    result = MarkdownNotesAdapter(path=str(tmp_path)).ingest()

    assert [unit.title for unit in result.units] == ["Heading Title", "plain-note"]
    assert [unit.metadata["source_file"] for unit in result.units] == [
        "has-heading.md",
        "plain-note.md",
    ]
    assert all(unit.metadata["frontmatter"] == {} for unit in result.units)


def test_markdown_notes_invalid_yaml_falls_back_to_raw_content(tmp_path):
    note = tmp_path / "broken.md"
    note.write_text(
        "---\n"
        "title: [broken\n"
        "---\n"
        "# Body Heading\n\n"
        "Still imported.\n",
        encoding="utf-8",
    )

    unit = MarkdownNotesAdapter(path=str(note)).ingest().units[0]

    assert unit.title == "Body Heading"
    assert unit.content.startswith("---\ntitle: [broken\n---\n# Body Heading")
    assert unit.metadata["frontmatter"] == {}
    assert "frontmatter_parse_error" in unit.metadata


def test_markdown_notes_normalizes_tags_from_lists_and_delimited_strings(tmp_path):
    list_note = tmp_path / "a.md"
    string_note = tmp_path / "b.md"
    list_note.write_text(
        "---\ntags: ['#Alpha', beta, Alpha, 'gamma; Delta']\n---\nA\n",
        encoding="utf-8",
    )
    string_note.write_text(
        "---\ntags: '#One, Two | three; one'\n---\nB\n",
        encoding="utf-8",
    )

    result = MarkdownNotesAdapter(path=str(tmp_path)).ingest()

    assert [unit.tags for unit in result.units] == [
        ["alpha", "beta", "gamma", "delta"],
        ["one", "two", "three"],
    ]


def test_markdown_notes_extracts_unique_wikilinks(tmp_path):
    note = tmp_path / "links.md"
    note.write_text(
        "See [[Target]], [[Target]], [[Target#Part]], [[Other|alias]], and [[]].\n",
        encoding="utf-8",
    )

    unit = MarkdownNotesAdapter(path=str(note)).ingest().units[0]

    assert unit.metadata["wikilinks"] == [
        {"raw": "Target", "target": "Target"},
        {"raw": "Target#Part", "target": "Target", "heading": "Part"},
        {"raw": "Other|alias", "target": "Other", "alias": "alias"},
    ]


def test_markdown_notes_adapter_is_registered():
    assert "markdown_notes" in list_adapters()
    adapter = get_adapter("markdown_notes", path="/tmp/notes")
    assert isinstance(adapter, MarkdownNotesAdapter)
    assert adapter.name == "markdown_notes"
