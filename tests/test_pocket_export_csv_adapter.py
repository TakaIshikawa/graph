from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.pocket_export_csv import PocketExportCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_pocket_export_csv_imports_active_tagged_item(tmp_path):
    path = tmp_path / "pocket.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["title", "url", "time_added", "tags", "status", "favorite", "excerpt"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "title": "Useful Article",
                "url": "https://example.com/useful",
                "time_added": "2025-01-02T03:04:05Z",
                "tags": "#AI, Reading | ai",
                "status": "0",
                "favorite": "0",
                "excerpt": "A useful summary",
            }
        )

    result = PocketExportCsvAdapter(path=str(path)).ingest()
    second = PocketExportCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "pocket_export_csv"
    assert unit.source_entity_type == "saved_item"
    assert unit.source_id.startswith("pocket_export_csv:")
    assert unit.source_id == second.units[0].source_id
    assert unit.title == "Useful Article"
    assert unit.content == (
        "Useful Article\n"
        "URL: https://example.com/useful\n"
        "Status: active\n"
        "Tags: ai, reading\n"
        "Excerpt: A useful summary"
    )
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.tags == ["ai", "reading"]
    assert unit.metadata["source_url"] == "https://example.com/useful"
    assert unit.metadata["external_url"] == "https://example.com/useful"
    assert unit.metadata["status"] == "active"
    assert unit.metadata["favorite"] is False
    assert unit.metadata["archived"] is False
    assert unit.metadata["read"] is False
    assert unit.metadata["tags"] == ["ai", "reading"]
    assert unit.metadata["time_added"] == "2025-01-02T03:04:05Z"
    assert unit.metadata["added_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["source_file"] == "pocket.csv"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_pocket_export_csv_imports_archived_and_favorite_rows(tmp_path):
    path = tmp_path / "pocket.csv"
    path.write_text(
        "title,url,time_added,status,favorite,tags\n"
        "Archived,https://example.com/archive,1735689600,archive,false,read\n"
        "Favorite,https://example.com/favorite,1735776000,active,1,favs\n",
        encoding="utf-8",
    )

    result = PocketExportCsvAdapter(path=str(path)).ingest()

    archived = result.units[0]
    favorite = result.units[1]
    assert archived.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert archived.metadata["status"] == "archived"
    assert archived.metadata["archived"] is True
    assert archived.metadata["read"] is True
    assert archived.metadata["favorite"] is False
    assert favorite.created_at == datetime(2025, 1, 2, tzinfo=timezone.utc)
    assert favorite.metadata["status"] == "active"
    assert favorite.metadata["favorite"] is True
    assert favorite.metadata["archived"] is False
    assert "Favorite: true" in favorite.content


def test_pocket_export_csv_marks_read_rows(tmp_path):
    path = tmp_path / "pocket.csv"
    path.write_text(
        "title,url,time_added,status,favorite\n"
        "Read,https://example.com/read,2025-01-04T00:00:00Z,read,0\n",
        encoding="utf-8",
    )

    unit = PocketExportCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.metadata["status"] == "read"
    assert unit.metadata["read"] is True
    assert unit.metadata["archived"] is False


def test_pocket_export_csv_handles_sparse_rows_without_malformed_metadata(tmp_path):
    path = tmp_path / "pocket.csv"
    path.write_text("url\nhttps://example.com/sparse\n", encoding="utf-8")

    result = PocketExportCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "https://example.com/sparse"
    assert unit.content == "https://example.com/sparse\nURL: https://example.com/sparse"
    assert unit.metadata["title"] == "https://example.com/sparse"
    assert unit.metadata["url"] == "https://example.com/sparse"
    assert unit.metadata["status"] == ""
    assert unit.metadata["favorite"] is False
    assert unit.metadata["archived"] is False
    assert unit.metadata["read"] is False
    assert unit.metadata["tags"] == []
    assert "time_added" not in unit.metadata
    assert "added_at" not in unit.metadata
    assert "excerpt" not in unit.metadata


def test_pocket_export_csv_filters_since_and_entity_type(tmp_path):
    path = tmp_path / "pocket.csv"
    path.write_text(
        "title,url,time_added\n"
        "Old,https://example.com/old,2025-01-01T00:00:00Z\n"
        "New,https://example.com/new,2025-01-03T00:00:00Z\n",
        encoding="utf-8",
    )
    since = SyncState(
        source_project="pocket_export_csv",
        source_entity_type="saved_item",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    filtered = PocketExportCsvAdapter(path=str(path)).ingest(since=since)
    wrong_entity = PocketExportCsvAdapter(path=str(path)).ingest(entity_types=["archive"])

    assert [unit.title for unit in filtered.units] == ["New"]
    assert wrong_entity.units == []
    assert wrong_entity.edges == []
