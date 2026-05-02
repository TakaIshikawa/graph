from __future__ import annotations

from datetime import datetime, timedelta, timezone

from graph.adapters.notion_markdown import NotionMarkdownAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_notion_markdown_ingests_frontmatter_properties(tmp_path):
    page = tmp_path / "Research" / "AI Strategy.md"
    page.parent.mkdir()
    page.write_text(
        """---
Tags:
  - AI
  - migration
Status: In Progress
Owner: Taka
---
# AI Strategy

Notes from the migration plan.
""",
        encoding="utf-8",
    )

    result = NotionMarkdownAdapter(path=str(tmp_path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.NOTION_MARKDOWN
    assert unit.source_id == "notion_markdown:Research/AI Strategy.md"
    assert unit.source_entity_type == "notion_page"
    assert unit.title == "AI Strategy"
    assert unit.content == "# AI Strategy\n\nNotes from the migration plan.\n"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.tags == ["AI", "migration", "status/in-progress"]
    assert unit.metadata == {
        "path": "Research/AI Strategy.md",
        "properties": {
            "Tags": ["AI", "migration"],
            "Status": "In Progress",
            "Owner": "Taka",
        },
        "raw_property_block": "Tags:\n  - AI\n  - migration\nStatus: In Progress\nOwner: Taka",
        "has_property_block": True,
    }


def test_notion_markdown_ingests_notion_key_value_block_after_heading(tmp_path):
    page = tmp_path / "Product Roadmap.md"
    page.write_text(
        """# Product Roadmap

Tags: planning, [[Product]]
Status: Published
Created: 2026-01-10

## Next

Ship the importer.
""",
        encoding="utf-8",
    )

    result = NotionMarkdownAdapter(path=str(page)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_id == "notion_markdown:Product Roadmap.md"
    assert unit.title == "Product Roadmap"
    assert unit.tags == ["planning", "Product", "status/published"]
    assert unit.metadata["properties"] == {
        "Tags": "planning, [[Product]]",
        "Status": "Published",
        "Created": "2026-01-10",
    }
    assert unit.metadata["raw_property_block"] == (
        "Tags: planning, [[Product]]\nStatus: Published\nCreated: 2026-01-10"
    )
    assert "Tags: planning" not in unit.content
    assert unit.content.startswith("# Product Roadmap\n")
    assert "Ship the importer." in unit.content


def test_notion_markdown_without_property_block_still_ingests(tmp_path):
    page = tmp_path / "Loose Note.md"
    page.write_text(
        """Intro paragraph.

## Details

No page properties here.
""",
        encoding="utf-8",
    )

    result = NotionMarkdownAdapter(path=str(page)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Details"
    assert unit.content.startswith("Intro paragraph.")
    assert unit.tags == []
    assert unit.metadata["properties"] == {}
    assert unit.metadata["raw_property_block"] == ""
    assert unit.metadata["has_property_block"] is False


def test_notion_markdown_source_ids_are_stable_for_same_input_path(tmp_path):
    page = tmp_path / "Archive" / "Same.md"
    page.parent.mkdir()
    page.write_text("# Same\n", encoding="utf-8")

    first = NotionMarkdownAdapter(path=str(tmp_path)).ingest().units[0].source_id
    second = NotionMarkdownAdapter(path=str(tmp_path)).ingest().units[0].source_id

    assert first == second == "notion_markdown:Archive/Same.md"


def test_notion_markdown_filters_entity_types_and_since(tmp_path):
    old_page = tmp_path / "old.md"
    old_page.write_text("# Old\n", encoding="utf-8")

    empty = NotionMarkdownAdapter(path=str(tmp_path)).ingest(entity_types=["markdown_note"])
    assert empty.units == []

    future = SyncState(
        source_project="notion_markdown",
        source_entity_type="notion_page",
        last_sync_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    skipped = NotionMarkdownAdapter(path=str(tmp_path)).ingest(since=future)
    assert skipped.units == []


def test_notion_markdown_adapter_is_registered():
    assert "notion_markdown" in list_adapters()
    adapter = get_adapter("notion_markdown", path="/tmp/notion")
    assert isinstance(adapter, NotionMarkdownAdapter)
    assert adapter.name == "notion_markdown"
