from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.libby_reading_activity_csv import LibbyReadingActivityCsvAdapter
from graph.types.models import SyncState


def test_libby_reading_activity_csv_ingests_action_metadata_and_raw_record(tmp_path):
    export = tmp_path / "activity.csv"
    export.write_text(
        "Loan ID,Title,Authors,Format,Library,Card,Activity,Borrowed Date,Due Date,Returned Date\n"
        "loan-1,The Book,\"Ada Lovelace; Grace Hopper\",eBook,City Library,Main Card,Return,2026-05-01,2026-05-21,2026-05-10\n",
        encoding="utf-8",
    )

    unit = LibbyReadingActivityCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_project == "libby_reading_activity_csv"
    assert unit.source_entity_type == "library_activity"
    assert unit.title == "Returned: The Book"
    assert unit.metadata["title"] == "The Book"
    assert unit.metadata["authors"] == ["Ada Lovelace", "Grace Hopper"]
    assert unit.metadata["format"] == "eBook"
    assert unit.metadata["library"] == "City Library"
    assert unit.metadata["card"] == "Main Card"
    assert unit.metadata["action"] == "return"
    assert unit.metadata["borrowed_at"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["due_at"] == "2026-05-21T00:00:00+00:00"
    assert unit.metadata["returned_at"] == "2026-05-10T00:00:00+00:00"
    assert unit.metadata["activity_at"] == "2026-05-10T00:00:00+00:00"
    assert unit.metadata["source_file"] == "activity.csv"
    assert unit.metadata["row_index"] == 1
    assert unit.metadata["raw_record"]["Loan ID"] == "loan-1"
    assert unit.created_at == datetime(2026, 5, 10, tzinfo=timezone.utc)


def test_libby_reading_activity_csv_directory_empty_rows_dedupes_and_filters(tmp_path):
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "activity.csv").write_text(
        "ID,Name,Creator,Media Format,Library Name,Action,Activity Date\n"
        "event-1,Old Book,Ada,Audiobook,City,Borrow,2026-04-01\n"
        ",,,,,,\n"
        "event-1,Old Book duplicate,Ada,Audiobook,City,Borrow,2026-04-01\n"
        "event-2,New Book,Grace,Kindle,County,Hold,2026-05-05\n",
        encoding="utf-8",
    )

    adapter = LibbyReadingActivityCsvAdapter(path=str(tmp_path))
    since = SyncState(
        source_project="libby_reading_activity_csv",
        source_entity_type="library_activity",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )
    result = adapter.ingest(since=since)

    assert [unit.title for unit in result.units] == ["Held: New Book"]
    assert result.units[0].metadata["author"] == "Grace"
    assert adapter.ingest(entity_types=["loan"]).units == []


def test_libby_reading_activity_csv_id_uses_action_and_date_with_fallback_digest(tmp_path):
    export = tmp_path / "activity.csv"
    export.write_text(
        "Loan ID,Title,Action,Activity Date\n"
        "loan-1,The Book,Borrow,2026-05-01\n"
        "loan-1,The Book,Return,2026-05-10\n"
        ",Fallback Book,Borrow,2026-05-02\n",
        encoding="utf-8",
    )

    units = LibbyReadingActivityCsvAdapter(path=str(export)).ingest().units
    repeated = LibbyReadingActivityCsvAdapter(path=str(export)).ingest().units

    assert len({unit.source_id for unit in units}) == 3
    assert [unit.source_id for unit in units] == [unit.source_id for unit in repeated]
    assert all(unit.source_id.startswith("libby_reading_activity_csv:") for unit in units)
