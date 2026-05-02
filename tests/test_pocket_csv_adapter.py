from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.pocket_csv import PocketCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_pocket_csv_ingests_minimal_row_with_url_source_id(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "title,url\n"
        "Example Article,https://example.com/articles/1\n",
        encoding="utf-8",
    )

    first = PocketCsvAdapter(path=str(export)).ingest()
    second = PocketCsvAdapter(path=str(export)).ingest()

    assert len(first.units) == 1
    unit = first.units[0]
    assert unit.source_project == SourceProject.POCKET_CSV
    assert unit.source_entity_type == "saved_item"
    assert unit.source_id == "url:https://example.com/articles/1"
    assert unit.source_id == second.units[0].source_id
    assert unit.title == "Example Article"
    assert unit.content == "Example Article\nURL: https://example.com/articles/1"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.metadata["title"] == "Example Article"
    assert unit.metadata["url"] == "https://example.com/articles/1"
    assert unit.metadata["tags"] == []


def test_pocket_csv_handles_quoted_commas_in_title_and_excerpt(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "title,url,excerpt,tags\n"
        '"Research, Notes",https://example.com/notes,"A summary, with commas","AI, Reading"\n',
        encoding="utf-8",
    )

    result = PocketCsvAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.title == "Research, Notes"
    assert unit.metadata["excerpt"] == "A summary, with commas"
    assert "Excerpt: A summary, with commas" in unit.content
    assert unit.tags == ["ai", "reading"]


def test_pocket_csv_splits_and_normalizes_tags(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "title,url,tags\n"
        'Tagged,https://example.com/tagged,"#AI; Reading | ai, Long Form"\n',
        encoding="utf-8",
    )

    result = PocketCsvAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.tags == ["ai", "reading", "long form"]
    assert unit.metadata["tags"] == ["ai", "reading", "long form"]


def test_pocket_csv_missing_optional_columns_do_not_fail(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "url\n"
        "https://example.com/untitled\n",
        encoding="utf-8",
    )

    result = PocketCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "https://example.com/untitled"
    assert unit.metadata["status"] == ""
    assert unit.metadata["archived"] is False
    assert unit.metadata["favorite"] is False
    assert unit.metadata["read"] is False
    assert unit.metadata["excerpt"] == ""


def test_pocket_csv_status_and_time_metadata(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "title,url,time_added,status,favorite,time_read\n"
        "Archived,https://example.com/archive,1704067200,archive,1,1704153600\n",
        encoding="utf-8",
    )

    result = PocketCsvAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.created_at == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert unit.metadata["time_added"] == "1704067200"
    assert unit.metadata["status"] == "archive"
    assert unit.metadata["archived"] is True
    assert unit.metadata["favorite"] is True
    assert unit.metadata["read"] is True


def test_pocket_csv_since_filter_and_entity_filter(tmp_path):
    export = tmp_path / "pocket.csv"
    export.write_text(
        "title,url,time_added\n"
        "Old,https://example.com/old,2024-01-01T00:00:00Z\n"
        "New,https://example.com/new,2024-01-03T00:00:00Z\n",
        encoding="utf-8",
    )

    filtered = PocketCsvAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="pocket_csv",
            source_entity_type="saved_item",
            last_sync_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    )
    wrong_entity = PocketCsvAdapter(path=str(export)).ingest(entity_types=["article"])

    assert [unit.title for unit in filtered.units] == ["New"]
    assert wrong_entity.units == []
    assert wrong_entity.edges == []


def test_pocket_csv_adapter_is_registered():
    assert "pocket_csv" in list_adapters()
    adapter = get_adapter("pocket_csv", path="/tmp/pocket.csv")
    assert isinstance(adapter, PocketCsvAdapter)
    assert adapter.name == "pocket_csv"
