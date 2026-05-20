from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.readwise_reader_documents_csv import ReadwiseReaderDocumentsCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_readwise_reader_documents_csv_ingests_document_metadata(tmp_path):
    export = tmp_path / "documents.csv"
    _write_csv(
        export,
        [
            {
                "Document ID": "doc-123",
                "Title": "Long Article",
                "URL": "https://example.com/article",
                "Author": "Ada Lovelace",
                "Source": "Example",
                "Category": "article",
                "Location": "Archive",
                "Tags": "#ai, research",
                "Saved At": "2026-05-01T10:00:00Z",
                "Last Opened At": "2026-05-02T12:30:00Z",
                "Reading Progress": "75%",
            }
        ],
    )

    result = ReadwiseReaderDocumentsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "readwise_reader_documents_csv"
    assert unit.source_id == "readwise_reader_documents_csv:doc-123"
    assert unit.source_entity_type == "reader_document"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.title == "Long Article"
    assert "Author: Ada Lovelace" in unit.content
    assert "URL: https://example.com/article" in unit.content
    assert unit.metadata["document_id"] == "doc-123"
    assert unit.metadata["url"] == "https://example.com/article"
    assert unit.metadata["author"] == "Ada Lovelace"
    assert unit.metadata["source"] == "Example"
    assert unit.metadata["category"] == "article"
    assert unit.metadata["location"] == "Archive"
    assert unit.metadata["tags"] == ["ai", "research"]
    assert unit.metadata["reading_progress"] == 75.0
    assert unit.metadata["source_file"] == "documents.csv"
    assert unit.metadata["source_row"]["Title"] == "Long Article"
    assert unit.created_at == datetime(2026, 5, 1, 10, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 2, 12, 30, tzinfo=timezone.utc)


def test_readwise_reader_documents_csv_uses_stable_fallback_id(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    row = {"Title": "Stable article", "URL": "https://example.com/stable", "Saved At": "2026-05-01T00:00:00Z"}
    _write_csv(first, [row])
    _write_csv(second, [row])

    first_unit = ReadwiseReaderDocumentsCsvAdapter(path=str(first)).ingest().units[0]
    second_unit = ReadwiseReaderDocumentsCsvAdapter(path=str(second)).ingest().units[0]

    assert first_unit.source_id == second_unit.source_id
    assert first_unit.source_id.startswith("readwise_reader_documents_csv:")


def test_readwise_reader_documents_csv_directory_skips_bad_files_dedupes_and_sorts(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    bad = tmp_path / "bad.csv"
    _write_csv(
        first,
        [
            {"Document ID": "2", "Title": "Second", "Saved At": "2026-05-02T00:00:00Z"},
            {"Document ID": "1", "Title": "First", "Saved At": "2026-05-01T00:00:00Z"},
        ],
    )
    _write_csv(second, [{"Document ID": "2", "Title": "Second updated", "Last Opened At": "2026-05-03T00:00:00Z"}])
    bad.write_bytes(b"\xff\xfe\x00")

    result = ReadwiseReaderDocumentsCsvAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == ["readwise_reader_documents_csv:1", "readwise_reader_documents_csv:2"]
    assert [unit.title for unit in result.units] == ["First", "Second updated"]
    assert [(unit.updated_at, unit.source_id) for unit in result.units] == sorted((unit.updated_at, unit.source_id) for unit in result.units)


def test_readwise_reader_documents_csv_filters_since_and_entity_type(tmp_path):
    export = tmp_path / "documents.csv"
    _write_csv(
        export,
        [
            {"Document ID": "1", "Title": "Old", "Last Opened At": "2026-05-01T00:00:00Z"},
            {"Document ID": "2", "Title": "Boundary", "Last Opened At": "2026-05-02T00:00:00Z"},
            {"Document ID": "3", "Title": "New", "Last Opened At": "2026-05-03T00:00:00Z"},
        ],
    )
    since = SyncState(
        source_project="readwise_reader_documents_csv",
        source_entity_type="reader_document",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    skipped = ReadwiseReaderDocumentsCsvAdapter(path=str(export)).ingest(entity_types=["highlight"])
    result = ReadwiseReaderDocumentsCsvAdapter(path=str(export)).ingest(since=since)

    assert skipped.units == []
    assert [unit.title for unit in result.units] == ["New"]
