from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.pocket_export_csv import PocketExportCsvAdapter
from graph.types.enums import ContentType, EdgeRelation, EdgeSource
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

    saved_items = [unit for unit in result.units if unit.source_entity_type == "saved_item"]
    assert len(saved_items) == 1
    unit = saved_items[0]
    assert unit.source_project == "pocket_export_csv"
    assert unit.source_entity_type == "saved_item"
    assert unit.source_id.startswith("pocket_export_csv:")
    assert unit.source_id == next(item for item in second.units if item.source_entity_type == "saved_item").source_id
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

    result = PocketExportCsvAdapter(path=str(path)).ingest(entity_types=["saved_item"])

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

    unit = PocketExportCsvAdapter(path=str(path)).ingest(entity_types=["saved_item"]).units[0]

    assert unit.metadata["status"] == "read"
    assert unit.metadata["read"] is True
    assert unit.metadata["archived"] is False


def test_pocket_export_csv_handles_sparse_rows_without_malformed_metadata(tmp_path):
    path = tmp_path / "pocket.csv"
    path.write_text("url\nhttps://example.com/sparse\n", encoding="utf-8")

    result = PocketExportCsvAdapter(path=str(path)).ingest()

    saved_items = [unit for unit in result.units if unit.source_entity_type == "saved_item"]
    assert len(saved_items) == 1
    unit = saved_items[0]
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

    assert [unit.title for unit in filtered.units if unit.source_entity_type == "saved_item"] == ["New"]
    assert wrong_entity.units == []
    assert wrong_entity.edges == []


def test_pocket_export_csv_emits_domain_units_and_edges(tmp_path):
    path = tmp_path / "pocket.csv"
    path.write_text(
        "title,url,time_added,status,tags\n"
        "One,https://www.Example.com/one,2025-01-01T00:00:00Z,active,ai\n"
        "Two,https://example.com/two,2025-01-02T00:00:00Z,archive,reading|ai\n"
        "Other,https://docs.example.com/three,2025-01-03T00:00:00Z,read,docs\n"
        "Bad,not a url,2025-01-04T00:00:00Z,active,bad\n",
        encoding="utf-8",
    )

    result = PocketExportCsvAdapter(path=str(path)).ingest()

    saved_items = [unit for unit in result.units if unit.source_entity_type == "saved_item"]
    domains = sorted((unit for unit in result.units if unit.source_entity_type == "domain"), key=lambda unit: unit.title)
    assert [unit.title for unit in domains] == ["docs.example.com", "example.com"]
    example = next(unit for unit in domains if unit.title == "example.com")
    assert example.metadata["domain"] == "example.com"
    assert example.metadata["item_count"] == 2
    assert example.metadata["saved_item_source_ids"] == sorted(
        item.source_id for item in saved_items if item.metadata["domain"] == "example.com"
    )
    assert example.metadata["statuses"] == ["active", "archived"]
    assert example.metadata["tags"] == ["ai", "reading"]
    assert example.metadata["source_files"] == ["pocket.csv"]
    assert len(result.edges) == 3
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}
    assert {edge.to_unit_id for edge in result.edges} == {
        item.source_id for item in saved_items if item.metadata["domain"]
    }


def test_pocket_export_csv_domain_filtering(tmp_path):
    path = tmp_path / "pocket.csv"
    path.write_text("title,url\nOne,https://example.com/one\n", encoding="utf-8")

    domain_only = PocketExportCsvAdapter(path=str(path)).ingest(entity_types=["domain"])
    item_only = PocketExportCsvAdapter(path=str(path)).ingest(entity_types=["saved_item"])

    assert [unit.source_entity_type for unit in domain_only.units] == ["domain"]
    assert domain_only.edges == []
    assert [unit.source_entity_type for unit in item_only.units] == ["saved_item"]
    assert item_only.edges == []
