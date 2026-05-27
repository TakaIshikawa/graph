from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.notion_database_csv import NotionDatabaseCsvAdapter
from graph.types.models import SyncState


def test_notion_database_csv_preserves_properties_and_tags(tmp_path):
    export = tmp_path / "notion.csv"
    export.write_text(
        "Name,URL,Created time,Last edited time,Tags,Status,Description\nRoadmap,https://notion.so/a,2024-01-01T00:00:00Z,2024-01-02T00:00:00Z,\"Product, Planning\",Active,Plan body\n",
        encoding="utf-8",
    )

    unit = NotionDatabaseCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Roadmap"
    assert unit.metadata["properties"]["Status"] == "Active"
    assert unit.metadata["tags"] == ["Product", "Planning"]
    assert unit.tags == ["Product", "Planning"]
    assert "Plan body" in unit.content


def test_notion_database_csv_supports_since_filter(tmp_path):
    export = tmp_path / "notion.csv"
    export.write_text("Title,Created time,Last edited time\nOld,2024-01-01T00:00:00Z,2024-01-02T00:00:00Z\n", encoding="utf-8")

    result = NotionDatabaseCsvAdapter(path=str(export)).ingest(since=SyncState(source_project="notion_database_csv", source_entity_type="page", last_sync_at=datetime(2024, 2, 1, tzinfo=timezone.utc)))

    assert result.units == []
