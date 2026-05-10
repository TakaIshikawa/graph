from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from graph.adapters.notion_export import NotionExportAdapter
from graph.types.enums import ContentType, EdgeRelation, SourceProject


def test_notion_export_parses_html_page(tmp_path):
    page = tmp_path / "My Page abc123.html"
    page.write_text(
        """
        <html>
        <head><title>My Page</title></head>
        <body>
            <h1>My Page</h1>
            <p>This is the content of my Notion page.</p>
            <div class="property">
                <span>Status</span>: <span>In Progress</span>
            </div>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = NotionExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.NOTION_EXPORT
    assert unit.source_entity_type == "page"
    assert unit.title == "My Page"
    assert "My Page" in unit.content
    assert "This is the content" in unit.content
    assert unit.content_type == ContentType.ARTIFACT
    assert "path" in unit.metadata


def test_notion_export_parses_markdown_page(tmp_path):
    page = tmp_path / "Meeting Notes def456.md"
    page.write_text(
        """# Meeting Notes

Date: 2026-05-10
Tags: work, meetings

Discussed project roadmap and next steps.
""",
        encoding="utf-8",
    )

    result = NotionExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.NOTION_EXPORT
    assert unit.source_entity_type == "page"
    assert unit.title == "Meeting Notes"
    assert "Meeting Notes" in unit.content
    assert "project roadmap" in unit.content
    assert unit.metadata.get("properties", {}).get("Date") == "2026-05-10"
    assert unit.metadata.get("properties", {}).get("Tags") == "work, meetings"


def test_notion_export_extracts_hierarchical_relationships(tmp_path):
    # Create parent page (in root)
    parent_page = tmp_path / "Parent Page xyz789.md"
    parent_page.write_text("# Parent Page\n\nParent content", encoding="utf-8")

    # Create subdirectory for nested pages
    parent_dir = tmp_path / "Parent Page xyz789"
    parent_dir.mkdir()

    # Create child page (in subdirectory)
    child_page = parent_dir / "Child Page abc123.md"
    child_page.write_text("# Child Page\n\nChild content", encoding="utf-8")

    result = NotionExportAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 2
    assert len(result.edges) == 1

    # Verify edge represents parent-child relationship
    edge = result.edges[0]
    assert edge.relation == EdgeRelation.CONTAINS
    assert "page" in edge.from_unit_id
    assert "page" in edge.to_unit_id


def test_notion_export_handles_zip_archive(tmp_path):
    # Create a ZIP file with Notion exports
    zip_path = tmp_path / "notion_export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("Page 1.md", "# Page 1\n\nContent 1")
        zf.writestr("Page 2.html", "<html><body><h1>Page 2</h1><p>Content 2</p></body></html>")

    result = NotionExportAdapter(path=str(zip_path)).ingest()

    assert len(result.units) == 2
    titles = {unit.title for unit in result.units}
    assert "Page 1" in titles
    assert "Page 2" in titles


def test_notion_export_extracts_media_references(tmp_path):
    page = tmp_path / "Page with Images.html"
    page.write_text(
        """
        <html>
        <body>
            <h1>Page with Images</h1>
            <img src="image1.png" />
            <img src="diagrams/flow.jpg" />
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = NotionExportAdapter(path=str(tmp_path)).ingest()

    unit = result.units[0]
    assert "media_files" in unit.metadata
    media_files = unit.metadata["media_files"]
    assert "image1.png" in media_files
    assert "diagrams/flow.jpg" in media_files


def test_notion_export_detects_database_view(tmp_path):
    db_page = tmp_path / "Task Database.html"
    db_page.write_text(
        """
        <html>
        <body>
            <div class="notion-database">
                <table>
                    <tr><th>Task</th><th>Status</th></tr>
                    <tr><td>Review PR</td><td>Done</td></tr>
                </table>
            </div>
        </body>
        </html>
        """,
        encoding="utf-8",
    )

    result = NotionExportAdapter(path=str(tmp_path)).ingest()

    unit = result.units[0]
    assert unit.source_entity_type == "database"
    assert unit.metadata.get("is_database") is True


def test_notion_export_filters_by_entity_type(tmp_path):
    (tmp_path / "Regular Page.md").write_text("# Regular Page\n\nContent", encoding="utf-8")
    db_page = tmp_path / "Database.html"
    db_page.write_text(
        '<html><body><div class="notion-database"><table></table></div></body></html>',
        encoding="utf-8",
    )

    # Ingest only pages
    result_pages = NotionExportAdapter(path=str(tmp_path)).ingest(entity_types=["page"])
    assert len(result_pages.units) == 1
    assert result_pages.units[0].source_entity_type == "page"

    # Ingest only databases
    result_dbs = NotionExportAdapter(path=str(tmp_path)).ingest(entity_types=["database"])
    assert len(result_dbs.units) == 1
    assert result_dbs.units[0].source_entity_type == "database"


def test_notion_export_extracts_tags_from_properties(tmp_path):
    page = tmp_path / "Tagged Page.md"
    page.write_text(
        """# Tagged Page

Tags: productivity, #work, learning
Category: Personal

Content here.
""",
        encoding="utf-8",
    )

    result = NotionExportAdapter(path=str(tmp_path)).ingest()

    unit = result.units[0]
    assert "productivity" in unit.tags
    assert "work" in unit.tags
    assert "learning" in unit.tags


def test_notion_export_empty_directory_returns_empty_result(tmp_path):
    result = NotionExportAdapter(path=str(tmp_path)).ingest()
    assert len(result.units) == 0
    assert len(result.edges) == 0


def test_notion_export_nonexistent_path_returns_empty_result(tmp_path):
    result = NotionExportAdapter(path=str(tmp_path / "nonexistent")).ingest()
    assert len(result.units) == 0


def test_notion_export_removes_uuid_from_filename(tmp_path):
    # Notion exports files with UUIDs appended
    page = tmp_path / "My Document 1a2b3c4d5e6f7g8h9i0j1k2l3m4n5o6p.md"
    page.write_text("# My Document\n\nContent", encoding="utf-8")

    result = NotionExportAdapter(path=str(tmp_path)).ingest()

    unit = result.units[0]
    assert unit.title == "My Document"
