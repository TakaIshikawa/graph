from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.notion_markdown_export import NotionMarkdownExportAdapter
from graph.types.models import SyncState


def test_notion_markdown_export_ingests_frontmatter_properties(tmp_path):
    page = tmp_path / "Projects" / "Roadmap.md"
    page.parent.mkdir()
    page.write_text(
        """---
ID: page-123
Tags:
  - Product
  - Planning
Created Time: 2025-01-01T10:00:00Z
Last Edited Time: 2025-01-02T11:00:00Z
Source URL: https://www.notion.so/page-123
---
# Roadmap

See [[Launch Plan]].
""",
        encoding="utf-8",
    )

    unit = NotionMarkdownExportAdapter(path=str(tmp_path)).ingest().units[0]

    assert unit.source_project == "notion_markdown_export"
    assert unit.source_entity_type == "page"
    assert unit.title == "Roadmap"
    assert unit.content == "# Roadmap\n\nSee [[Launch Plan]].\n"
    assert unit.tags == ["Product", "Planning"]
    assert unit.metadata["path"] == "Projects/Roadmap.md"
    assert unit.metadata["property_id"] == "page-123"
    assert unit.metadata["source_url"] == "https://www.notion.so/page-123"
    assert unit.metadata["backlinks"] == ["Launch Plan"]
    assert unit.created_at == datetime(2025, 1, 1, 10, 0, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, 11, 0, tzinfo=timezone.utc)


def test_notion_markdown_export_traverses_nested_dirs_and_preserves_property_body_boundary(tmp_path):
    first = tmp_path / "A" / "First.md"
    second = tmp_path / "B" / "Second.md"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("# First\n\nTags: alpha, [[Beta]]\nCreated: 2025-01-01\n\nBody text.\n", encoding="utf-8")
    second.write_text("No heading body.\n", encoding="utf-8")

    units = NotionMarkdownExportAdapter(path=str(tmp_path)).ingest().units

    assert [unit.metadata["path"] for unit in units] == ["A/First.md", "B/Second.md"]
    first_unit = units[0]
    assert first_unit.title == "First"
    assert first_unit.tags == ["alpha", "Beta"]
    assert "Tags: alpha" not in first_unit.content
    assert first_unit.content.startswith("# First")
    assert units[1].title == "Second"


def test_notion_markdown_export_filters_since_and_entity_type(tmp_path):
    page = tmp_path / "Page.md"
    page.write_text("---\nUpdated: 2025-01-01\n---\n# Page\n", encoding="utf-8")

    assert NotionMarkdownExportAdapter(path=str(tmp_path)).ingest(entity_types=["notion_page"]).units == []
    since = SyncState(
        source_project="notion_markdown_export",
        source_entity_type="page",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert NotionMarkdownExportAdapter(path=str(tmp_path)).ingest(since=since).units == []
