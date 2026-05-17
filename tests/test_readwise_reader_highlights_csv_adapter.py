from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.readwise_reader_highlights_csv import ReadwiseReaderHighlightsCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_readwise_reader_highlights_csv_ingests_highlights(tmp_path):
    export = tmp_path / "reader.csv"
    export.write_text(
        "Highlight ID,Highlight,Document,Author,URL,Note,Tags,Location,Highlighted At,Updated At\n"
        'abc-123,"Important passage",Long Article,Ada Lovelace,https://example.com/article,"Follow up","#ai, research",42,2026-05-01T10:00:00Z,2026-05-02T12:30:00Z\n',
        encoding="utf-8",
    )

    result = ReadwiseReaderHighlightsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "readwise_reader_highlights_csv"
    assert unit.source_id == "readwise_reader_highlights_csv:abc-123"
    assert unit.source_entity_type == "highlight"
    assert unit.content_type == ContentType.INSIGHT
    assert unit.title == "Long Article"
    assert unit.content.startswith("Important passage")
    assert "Note: Follow up" in unit.content
    assert unit.metadata["highlight"] == "Important passage"
    assert unit.metadata["document"] == "Long Article"
    assert unit.metadata["author"] == "Ada Lovelace"
    assert unit.metadata["url"] == "https://example.com/article"
    assert unit.metadata["note"] == "Follow up"
    assert unit.metadata["tags"] == ["ai", "research"]
    assert unit.metadata["location"] == "42"
    assert unit.created_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 2, 12, 30, tzinfo=timezone.utc)


def test_readwise_reader_highlights_csv_uses_digest_fallback_id(tmp_path):
    export = tmp_path / "reader.csv"
    export.write_text(
        "Highlight,Title,Author,Location,Highlighted At\n"
        "Stable highlight,Article,Grace Hopper,loc-7,2026-05-01T10:00:00Z\n",
        encoding="utf-8",
    )

    first = ReadwiseReaderHighlightsCsvAdapter(path=str(export)).ingest().units[0]
    second = ReadwiseReaderHighlightsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("readwise_reader_highlights_csv:")


def test_readwise_reader_highlights_csv_skips_blank_rows_and_sorts(tmp_path):
    export = tmp_path / "reader.csv"
    export.write_text(
        "Highlight,Document,Highlighted At\n"
        ",,\n"
        "Second,Doc,2026-05-02T00:00:00Z\n"
        "First,Doc,2026-05-01T00:00:00Z\n",
        encoding="utf-8",
    )

    result = ReadwiseReaderHighlightsCsvAdapter(path=str(export)).ingest()

    assert [(unit.created_at, unit.source_id) for unit in result.units] == sorted((unit.created_at, unit.source_id) for unit in result.units)
    assert [unit.metadata["highlight"] for unit in result.units] == ["First", "Second"]


def test_readwise_reader_highlights_csv_filters_since_by_updated_at(tmp_path):
    export = tmp_path / "reader.csv"
    export.write_text(
        "Highlight,Document,Highlighted At,Updated At\n"
        "Old,Doc,2026-05-01T00:00:00Z,2026-05-02T00:00:00Z\n"
        "New,Doc,2026-05-01T00:00:00Z,2026-05-03T00:00:00Z\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="readwise_reader_highlights_csv", source_entity_type="highlight", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    result = ReadwiseReaderHighlightsCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.metadata["highlight"] for unit in result.units] == ["New"]


def test_readwise_reader_highlights_csv_entity_types_filtering(tmp_path):
    export = tmp_path / "reader.csv"
    export.write_text("Highlight\nA passage\n", encoding="utf-8")

    assert ReadwiseReaderHighlightsCsvAdapter(path=str(export)).ingest(entity_types=["document"]).units == []
    assert len(ReadwiseReaderHighlightsCsvAdapter(path=str(export)).ingest(entity_types=["highlight"]).units) == 1
