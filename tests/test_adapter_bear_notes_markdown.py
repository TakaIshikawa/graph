from __future__ import annotations

from graph.adapters.bear_notes_markdown import BearNotesMarkdownAdapter


def test_bear_notes_markdown_ingests_nested_notes(tmp_path):
    note = tmp_path / "folder" / "note.md"
    note.parent.mkdir()
    note.write_text("---\ntitle: Frontmatter Title\ntags: work, Ideas\ncreated: 2024-01-01\n---\n# Heading\nBody #inline/tag", encoding="utf-8")

    unit = BearNotesMarkdownAdapter(path=str(tmp_path)).ingest().units[0]

    assert unit.title == "Frontmatter Title"
    assert unit.metadata["relative_path"] == "folder/note.md"
    assert unit.tags == ["work", "ideas", "inline/tag"]


def test_bear_notes_markdown_title_fallbacks(tmp_path):
    heading = tmp_path / "heading.md"
    filename = tmp_path / "filename.md"
    heading.write_text("# Heading Title\nBody", encoding="utf-8")
    filename.write_text("Body only", encoding="utf-8")

    titles = [unit.title for unit in BearNotesMarkdownAdapter(path=str(tmp_path)).ingest().units]

    assert sorted(titles) == ["Heading Title", "filename"]
