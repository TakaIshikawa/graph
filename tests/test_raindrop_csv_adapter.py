"""Tests for the Raindrop.io CSV bookmark export adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.raindrop_csv import RaindropCsvAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import SyncState


def test_raindrop_csv_ingests_normal_row_with_metadata(tmp_path):
    export = tmp_path / "raindrop.csv"
    export.write_text(
        "title,excerpt,note,url,folder,tags,created,domain\n"
        "Graph Notes,A useful graph article,Remember to cite this,"
        "https://example.com/graph,Reading,\"PKM, Research\","
        "2024-01-02T03:04:05Z,example.com\n",
        encoding="utf-8",
    )

    result = RaindropCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.RAINDROP_CSV
    assert unit.source_id == "url:https://example.com/graph"
    assert unit.source_entity_type == "bookmark"
    assert unit.title == "Graph Notes"
    assert unit.content == (
        "Graph Notes\n"
        "URL: https://example.com/graph\n"
        "Excerpt: A useful graph article\n"
        "Note: Remember to cite this\n"
        "Folder: Reading\n"
        "Domain: example.com\n"
        "Tags: pkm, research"
    )
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.tags == ["pkm", "research"]
    assert unit.metadata == {
        "title": "Graph Notes",
        "url": "https://example.com/graph",
        "excerpt": "A useful graph article",
        "note": "Remember to cite this",
        "folder": "Reading",
        "created_at": "2024-01-02T03:04:05Z",
        "updated_at": "",
        "domain": "example.com",
        "tags": ["pkm", "research"],
        "source_file": "raindrop.csv",
    }
    assert unit.created_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_raindrop_csv_blank_note_and_excerpt_still_ingest(tmp_path):
    export = tmp_path / "raindrop.csv"
    export.write_text(
        "title,excerpt,note,url,folder,tags\n"
        "Blank Optional,,,https://example.com/blank,Inbox,\n",
        encoding="utf-8",
    )

    result = RaindropCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Blank Optional"
    assert unit.content == "Blank Optional\nURL: https://example.com/blank\nFolder: Inbox"
    assert unit.metadata["excerpt"] == ""
    assert unit.metadata["note"] == ""
    assert unit.tags == []


def test_raindrop_csv_handles_quoted_commas_in_columns_and_tags(tmp_path):
    export = tmp_path / "raindrop.csv"
    export.write_text(
        "Title,URL,Excerpt,Note,Folder,Tags\n"
        '"Research, Notes",https://example.com/notes,'
        '"A summary, with commas","Keep this, especially","Reading, Later",'
        '"#AI; Long Form | reading, later"\n',
        encoding="utf-8",
    )

    result = RaindropCsvAdapter(path=str(export)).ingest()

    unit = result.units[0]
    assert unit.title == "Research, Notes"
    assert unit.metadata["excerpt"] == "A summary, with commas"
    assert unit.metadata["note"] == "Keep this, especially"
    assert unit.metadata["folder"] == "Reading, Later"
    assert unit.tags == ["ai", "long form", "reading", "later"]
    assert "Folder: Reading, Later" in unit.content


def test_raindrop_csv_stable_source_ids_use_url_or_row_content(tmp_path):
    export = tmp_path / "raindrop.csv"
    export.write_text(
        "title,excerpt,note,url\n"
        "With URL,Excerpt,Note,https://example.com/stable\n"
        "Without URL,Excerpt only,Note only,\n",
        encoding="utf-8",
    )

    first = RaindropCsvAdapter(path=str(export)).ingest()
    second = RaindropCsvAdapter(path=str(export)).ingest()

    assert [unit.source_id for unit in first.units] == [
        "raindrop_csv:f252072ee32fb0e653011fdf",
        "url:https://example.com/stable",
    ]
    assert [unit.source_id for unit in first.units] == [
        unit.source_id for unit in second.units
    ]


def test_raindrop_csv_since_filter_and_entity_filter(tmp_path):
    export = tmp_path / "raindrop.csv"
    export.write_text(
        "title,url,created\n"
        "Old,https://example.com/old,2024-01-01T00:00:00Z\n"
        "New,https://example.com/new,2024-01-03T00:00:00Z\n",
        encoding="utf-8",
    )

    filtered = RaindropCsvAdapter(path=str(export)).ingest(
        since=SyncState(
            source_project="raindrop_csv",
            source_entity_type="bookmark",
            last_sync_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    )
    wrong_entity = RaindropCsvAdapter(path=str(export)).ingest(entity_types=["article"])

    assert [unit.title for unit in filtered.units] == ["New"]
    assert wrong_entity.units == []
    assert wrong_entity.edges == []


def test_raindrop_csv_adapter_is_registered():
    assert "raindrop_csv" in list_adapters()
    adapter = get_adapter("raindrop_csv", path="/tmp/raindrop.csv")
    assert isinstance(adapter, RaindropCsvAdapter)
    assert adapter.name == "raindrop_csv"
