from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.openlibrary_reading_log_csv import OpenLibraryReadingLogCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_openlibrary_reading_log_csv_ingests_book_metadata_and_registry(tmp_path):
    export = tmp_path / "reading.csv"
    export.write_text(
        "\n".join(
            [
                "Title,Author,Shelf,ISBN,Work ID,Edition ID,Date Started,Date Finished,Rating,Notes",
                "The Book,Ada Lovelace,read,9781234567890,/works/OL1W,/books/OL2M,2025-01-01,2025-01-10,4.5,Great notes",
            ]
        ),
        encoding="utf-8",
    )

    result = OpenLibraryReadingLogCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.OPENLIBRARY_READING_LOG_CSV
    assert unit.source_entity_type == "book"
    assert unit.metadata["author"] == "Ada Lovelace"
    assert unit.metadata["shelf"] == "read"
    assert unit.metadata["isbn"] == "9781234567890"
    assert unit.metadata["work_id"] == "OL1W"
    assert unit.metadata["edition_id"] == "OL2M"
    assert unit.metadata["started_at"] == "2025-01-01T00:00:00+00:00"
    assert unit.metadata["finished_at"] == "2025-01-10T00:00:00+00:00"
    assert unit.metadata["rating"] == 4.5
    assert unit.metadata["notes"] == "Great notes"
    assert "Great notes" in unit.content
    assert unit.updated_at == datetime(2025, 1, 10, tzinfo=timezone.utc)
    assert get_adapter("openlibrary_reading_log_csv", path=str(export)).name == "openlibrary_reading_log_csv"


def test_openlibrary_reading_log_csv_aliases_directory_since_and_ids(tmp_path):
    (tmp_path / "one.csv").write_text("Book,Authors,Status,ISBN13,Started,Finished,Stars,Review\nOld,Ada,want,111,2025-01-01,2025-01-02,3,Old note\n", encoding="utf-8")
    (tmp_path / "two.csv").write_text("Name,Author,Read Status,OpenLibrary Work ID,Start Date,Date Read,My Rating,Comments\nNew,Grace,read,OL9W,2025-01-03,2025-01-04,5,New note\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = OpenLibraryReadingLogCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="openlibrary_reading_log_csv", source_entity_type="book", last_sync_at=datetime(2025, 1, 3, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert first.units[0].metadata["work_id"] == "OL9W"
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["author"]).units == []
