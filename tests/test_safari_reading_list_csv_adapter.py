from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.safari_reading_list_csv import SafariReadingListCsvAdapter
from graph.types.enums import SourceProject


def test_safari_reading_list_csv_ingests_read_and_unread_rows_and_registry(tmp_path):
    export = tmp_path / "reading-list.csv"
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Title", "URL", "Preview Text", "Added Date", "Read Status", "Folder", "Site Name"])
        writer.writeheader()
        writer.writerow({"Title": "Article", "URL": "https://example.com/a", "Preview Text": "Useful context", "Added Date": "2025-01-01T00:00:00Z", "Read Status": "read", "Folder": "Research", "Site Name": "Example"})
        writer.writerow({"Title": "Unread", "URL": "https://example.com/u", "Added Date": "2025-01-02T00:00:00Z", "Read Status": "unread"})

    result = SafariReadingListCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    unit = result.units[0]
    assert unit.source_project == SourceProject.SAFARI_READING_LIST_CSV
    assert unit.source_entity_type == "reading_list_entry"
    assert unit.title == "Article"
    assert unit.metadata["read"] is True
    assert unit.metadata["folder"] == "Research"
    assert unit.metadata["site_name"] == "Example"
    assert unit.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert "Useful context" in unit.content
    assert result.units[1].metadata["status"] == "unread"
    assert get_adapter("safari_reading_list_csv", path=str(export)).name == "safari_reading_list_csv"


def test_safari_reading_list_csv_missing_title_falls_back_to_site_or_url(tmp_path):
    export = tmp_path / "reading-list.csv"
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["URL", "Description", "Added Date"])
        writer.writeheader()
        writer.writerow({"URL": "https://www.example.org/path", "Description": "No explicit title", "Added Date": "2025-01-01"})

    unit = SafariReadingListCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "example.org"
    assert unit.metadata["preview_text"] == "No explicit title"
    assert unit.metadata["status"] == "unread"


def test_safari_reading_list_csv_duplicate_urls_with_different_dates_are_stable(tmp_path):
    export = tmp_path / "reading-list.csv"
    with export.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Title", "URL", "Added Date"])
        writer.writeheader()
        writer.writerow({"Title": "First", "URL": "https://example.com/a", "Added Date": "2025-01-01"})
        writer.writerow({"Title": "Second", "URL": "https://example.com/a", "Added Date": "2025-01-02"})

    adapter = SafariReadingListCsvAdapter(path=str(export))
    first = adapter.ingest()
    second = adapter.ingest()

    assert len({unit.source_id for unit in first.units}) == 2
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["bookmark"]).units == []
