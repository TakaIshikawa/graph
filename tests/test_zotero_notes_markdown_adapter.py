from __future__ import annotations

from graph.adapters.zotero_notes_markdown import ZoteroNotesMarkdownAdapter


def test_zotero_notes_markdown_ingests_single_file_frontmatter(tmp_path):
    note = tmp_path / "note.md"
    note.write_text("---\nitem_key: ABC123\ncitation_key: doe2024\ntags: research, notes\ncollections: Inbox; Papers\n---\n\nMarkdown body\n", encoding="utf-8")

    unit = ZoteroNotesMarkdownAdapter(path=str(note)).ingest().units[0]

    assert unit.metadata["item_key"] == "ABC123"
    assert unit.metadata["citation_key"] == "doe2024"
    assert unit.metadata["tags"] == ["research", "notes"]
    assert unit.metadata["collections"] == ["Inbox", "Papers"]
    assert unit.content == "Markdown body"


def test_zotero_notes_markdown_directory_and_empty_skip(tmp_path):
    (tmp_path / "empty.md").write_text("", encoding="utf-8")
    (tmp_path / "note.md").write_text("Body", encoding="utf-8")

    units = ZoteroNotesMarkdownAdapter(path=str(tmp_path)).ingest().units

    assert len(units) == 1
    assert units[0].content == "Body"
