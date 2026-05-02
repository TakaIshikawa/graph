from __future__ import annotations

from graph.adapters.markdown_links import MarkdownLinksAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject


def test_markdown_links_ingests_inline_links_with_context(tmp_path):
    note = tmp_path / "notes.md"
    note.write_text(
        "Read [Example Site](https://example.com/docs) for background.\n",
        encoding="utf-8",
    )

    result = MarkdownLinksAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.MARKDOWN_LINKS
    assert unit.source_entity_type == "markdown_link"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.title == "Example Site"
    assert unit.tags == ["markdown-link"]
    assert unit.metadata == {
        "url": "https://example.com/docs",
        "file_path": "notes.md",
        "line_number": 1,
        "link_text": "Example Site",
    }
    assert "URL: https://example.com/docs" in unit.content
    assert "File: notes.md:1" in unit.content
    assert "Read [Example Site](https://example.com/docs) for background." in unit.content


def test_markdown_links_ingests_reference_style_links_and_definitions(tmp_path):
    note = tmp_path / "refs.md"
    note.write_text(
        "\n".join(
            [
                "Use [Graph docs][docs] in the implementation.",
                "",
                "[docs]: https://example.org/graph \"Graph Docs\"",
            ]
        ),
        encoding="utf-8",
    )

    result = MarkdownLinksAdapter(path=str(note)).ingest()

    assert [(unit.metadata["line_number"], unit.metadata["link_text"]) for unit in result.units] == [
        (1, "Graph docs")
    ]
    unit = result.units[0]
    assert unit.metadata["url"] == "https://example.org/graph"
    assert unit.metadata["file_path"] == "refs.md"
    assert unit.title == "Graph docs"


def test_markdown_links_ingests_shortcut_reference_links(tmp_path):
    note = tmp_path / "shortcut.md"
    note.write_text(
        "Read [Docs] before shipping.\n\n[Docs]: <https://docs.example.com/start>\n",
        encoding="utf-8",
    )

    result = MarkdownLinksAdapter(path=str(note)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.metadata["url"] == "https://docs.example.com/start"
    assert unit.metadata["line_number"] == 1
    assert unit.metadata["link_text"] == "Docs"


def test_markdown_links_keeps_duplicate_url_occurrences_in_different_files(tmp_path):
    first = tmp_path / "a.md"
    second = tmp_path / "nested" / "b.md"
    second.parent.mkdir()
    first.write_text("[Shared](https://example.com/shared)\n", encoding="utf-8")
    second.write_text("[Shared](https://example.com/shared)\n", encoding="utf-8")

    result = MarkdownLinksAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert {unit.metadata["file_path"] for unit in result.units} == {"a.md", "nested/b.md"}
    assert len({unit.source_id for unit in result.units}) == 2


def test_markdown_links_ignores_images(tmp_path):
    note = tmp_path / "images.md"
    note.write_text(
        "![Alt text](https://example.com/image.png)\n[Page](https://example.com/page)\n",
        encoding="utf-8",
    )

    result = MarkdownLinksAdapter(path=str(tmp_path)).ingest()

    assert [unit.metadata["url"] for unit in result.units] == ["https://example.com/page"]


def test_markdown_links_adapter_is_registered():
    assert "markdown_links" in list_adapters()
    adapter = get_adapter("markdown_links", path="/tmp/notes")
    assert isinstance(adapter, MarkdownLinksAdapter)
    assert adapter.name == "markdown_links"
