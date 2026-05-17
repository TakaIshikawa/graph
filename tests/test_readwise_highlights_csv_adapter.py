from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.readwise_highlights_csv import ReadwiseHighlightsCsvAdapter
from graph.types.enums import ContentType


def test_readwise_highlights_csv_ingests_representative_highlight_row(tmp_path):
    export = tmp_path / "readwise.csv"
    export.write_text(
        (
            "Highlight ID,Highlight,Title,Author,Source Type,URL,Location,Note,Tags,Highlighted At,Updated At\n"
            'rw-1,"First line\nsecond line",Deep Work,Cal Newport,book,https://example.com/deep,42,'
            '"Remember this\nfor planning","focus, #productivity",2025-01-02T10:30:00Z,2025-01-03T11:45:00Z\n'
        ),
        encoding="utf-8",
    )

    result = ReadwiseHighlightsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "readwise_highlights_csv"
    assert unit.source_id == "readwise_highlights_csv:rw-1"
    assert unit.source_entity_type == "highlight"
    assert unit.content_type == ContentType.INSIGHT
    assert unit.title == "Deep Work"
    assert unit.content.startswith("First line\nsecond line")
    assert "Note: Remember this\nfor planning" in unit.content
    assert "Title: Deep Work" in unit.content
    assert "Author: Cal Newport" in unit.content
    assert "Source type: book" in unit.content
    assert "URL: https://example.com/deep" in unit.content
    assert "Location: 42" in unit.content
    assert unit.metadata["highlight"] == "First line\nsecond line"
    assert unit.metadata["title"] == "Deep Work"
    assert unit.metadata["author"] == "Cal Newport"
    assert unit.metadata["source_type"] == "book"
    assert unit.metadata["url"] == "https://example.com/deep"
    assert unit.metadata["location"] == "42"
    assert unit.metadata["note"] == "Remember this\nfor planning"
    assert unit.metadata["tags"] == ["focus", "productivity"]
    assert unit.metadata["highlighted_at"] == "2025-01-02T10:30:00+00:00"
    assert unit.metadata["updated_at"] == "2025-01-03T11:45:00+00:00"
    assert unit.tags == ["focus", "productivity"]
    assert unit.created_at == datetime(2025, 1, 2, 10, 30, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 3, 11, 45, tzinfo=timezone.utc)


def test_readwise_highlights_csv_handles_missing_optional_columns(tmp_path):
    export = tmp_path / "minimal.csv"
    export.write_text("Highlight\nA standalone highlight\n", encoding="utf-8")

    result = ReadwiseHighlightsCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.title == "Readwise highlight"
    assert unit.content == "A standalone highlight"
    assert unit.metadata["source_file"] == "minimal.csv"
    assert unit.metadata["row_number"] == 2
    assert unit.metadata["highlight"] == "A standalone highlight"
    assert "title" not in unit.metadata
    assert "url" not in unit.metadata
    assert unit.tags == []


def test_readwise_highlights_csv_source_ids_are_deterministic(tmp_path):
    export = tmp_path / "stable.csv"
    export.write_text(
        "Highlight,Title,Author,Location,Highlighted At\n"
        "Stable passage,Stable Book,Stable Author,7,2025-01-01T00:00:00Z\n",
        encoding="utf-8",
    )

    first = ReadwiseHighlightsCsvAdapter(path=str(export)).ingest().units[0]
    second = ReadwiseHighlightsCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id == second.source_id
    assert first.source_id.startswith("readwise_highlights_csv:")
    assert first.source_id != "readwise_highlights_csv:"
