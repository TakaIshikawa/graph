from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.markdown_frontmatter import MarkdownFrontmatterAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject


def test_markdown_frontmatter_ingests_known_fields_and_clean_body(tmp_path):
    root = tmp_path / "vault"
    notes = root / "notes"
    notes.mkdir(parents=True)
    note = notes / "strategy.md"
    note.write_text(
        "---\n"
        "title: Strategy Note\n"
        "tags:\n"
        "  - Product\n"
        "  - '#Research'\n"
        "created: 2026-01-02\n"
        "updated: 2026-01-03T04:05:06Z\n"
        "source_url: https://example.com/source\n"
        "aliases: [Plan A, Alternate Strategy]\n"
        "status: draft\n"
        "rating: 4\n"
        "---\n"
        "# Body Heading\n\n"
        "This is the note body.\n",
        encoding="utf-8",
    )

    result = MarkdownFrontmatterAdapter(
        path=str(root), source_id_root=str(root)
    ).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MARKDOWN_FRONTMATTER
    assert unit.source_entity_type == "markdown_frontmatter"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.title == "Strategy Note"
    assert unit.content == "# Body Heading\n\nThis is the note body.\n"
    assert "---" not in unit.content
    assert unit.tags == ["product", "research"]
    assert unit.created_at == datetime(2026, 1, 2, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 1, 3, 4, 5, 6, tzinfo=timezone.utc)
    assert unit.metadata["source_file"] == "notes/strategy.md"
    assert unit.metadata["source_url"] == "https://example.com/source"
    assert unit.metadata["aliases"] == ["Plan A", "Alternate Strategy"]
    assert unit.metadata["frontmatter_metadata"] == {"status": "draft", "rating": 4}
    assert unit.metadata["frontmatter"]["title"] == "Strategy Note"


def test_markdown_frontmatter_without_frontmatter_uses_heading_then_filename(tmp_path):
    heading_note = tmp_path / "has-heading.md"
    plain_note = tmp_path / "plain-note.markdown"
    heading_note.write_text("# Heading Title\n\nBody text\n", encoding="utf-8")
    plain_note.write_text("Body without a heading\n", encoding="utf-8")

    result = MarkdownFrontmatterAdapter(path=str(tmp_path)).ingest()

    assert [unit.title for unit in result.units] == ["Heading Title", "plain-note"]
    assert [unit.content for unit in result.units] == [
        "# Heading Title\n\nBody text\n",
        "Body without a heading\n",
    ]
    assert all(unit.metadata["frontmatter"] == {} for unit in result.units)
    assert all(unit.metadata["frontmatter_metadata"] == {} for unit in result.units)


def test_markdown_frontmatter_normalizes_tags_from_lists_and_delimited_strings(tmp_path):
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

    result = MarkdownFrontmatterAdapter(path=str(tmp_path)).ingest()

    assert [unit.tags for unit in result.units] == [
        ["alpha", "beta", "gamma", "delta"],
        ["one", "two", "three"],
    ]


def test_markdown_frontmatter_malformed_yaml_falls_back_to_raw_content(tmp_path):
    note = tmp_path / "broken.md"
    note.write_text(
        "---\n"
        "title: [broken\n"
        "---\n"
        "# Body Heading\n\n"
        "Still imported.\n",
        encoding="utf-8",
    )

    unit = MarkdownFrontmatterAdapter(path=str(note)).ingest().units[0]

    assert unit.title == "Body Heading"
    assert unit.content.startswith("---\ntitle: [broken\n---\n# Body Heading")
    assert unit.metadata["frontmatter"] == {}
    assert unit.metadata["frontmatter_metadata"] == {}
    assert "frontmatter_parse_error" in unit.metadata


def test_markdown_frontmatter_adapter_is_registered():
    assert "markdown_frontmatter" in list_adapters()
    adapter = get_adapter("markdown_frontmatter", path="/tmp/notes")
    assert isinstance(adapter, MarkdownFrontmatterAdapter)
    assert adapter.name == "markdown_frontmatter"
