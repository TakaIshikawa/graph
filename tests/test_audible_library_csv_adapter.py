from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.audible_library_csv import AudibleLibraryCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_audible_library_csv_ingests_audiobook_rows(tmp_path):
    export = tmp_path / "library.csv"
    _write_csv(
        export,
        [
            {
                "Title": "Example Audiobook",
                "Author": "A. Writer; B. Writer",
                "Narrator": "Voice One, Voice Two",
                "Purchase Date": "2026-05-01",
                "Release Date": "2025-12-31",
                "Duration": "10 hrs 30 mins",
                "Rating": "4.5",
                "ASIN": "B123",
                "Product URL": "https://www.audible.com/pd/B123",
            }
        ],
    )

    result = AudibleLibraryCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.AUDIBLE_LIBRARY_CSV
    assert unit.source_entity_type == "audiobook"
    assert unit.source_id == "audible_library_csv:asin:B123"
    assert unit.title == "Example Audiobook"
    assert unit.metadata["authors"] == ["A. Writer", "B. Writer"]
    assert unit.metadata["narrators"] == ["Voice One", "Voice Two"]
    assert unit.metadata["purchase_date"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["release_date"] == "2025-12-31T00:00:00+00:00"
    assert unit.metadata["duration_seconds"] == 36000
    assert unit.metadata["rating"] == 4.5
    assert unit.updated_at == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert "Narrator: Voice One, Voice Two" in unit.content


def test_audible_library_csv_directory_since_entity_filter_and_sparse_rows(tmp_path):
    _write_csv(
        tmp_path / "one.csv",
        [
            {"Title": "Old", "Purchase Date": "2026-04-30", "Product URL": "https://example.com/old"},
            {"Book Title": "New", "Date Purchased": "2026-05-03", "Length": "90 min"},
            {"Title": "", "Author": ""},
        ],
    )
    _write_csv(tmp_path / "two.csv", [{"Product Name": "Newest", "Updated At": "2026-05-04", "My Rating": "bad"}])
    since = SyncState(
        source_project="audible_library_csv",
        source_entity_type="audiobook",
        last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    result = AudibleLibraryCsvAdapter(path=str(tmp_path)).ingest(since=since)
    skipped = AudibleLibraryCsvAdapter(path=str(tmp_path)).ingest(entity_types=["book"])

    assert [unit.title for unit in result.units] == ["New", "Newest"]
    assert result.units[0].metadata["duration_seconds"] == 5400
    assert "rating" not in result.units[1].metadata
    assert skipped.units == []
    assert get_adapter("audible_library_csv", path=str(tmp_path)).name == "audible_library_csv"


def test_audible_library_csv_url_fallback_id_is_deterministic(tmp_path):
    export = tmp_path / "library.csv"
    _write_csv(export, [{"Title": "URL Identity", "URL": "https://audible.example/item"}])

    first = AudibleLibraryCsvAdapter(path=str(export)).ingest().units[0]
    second = AudibleLibraryCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_id.startswith("audible_library_csv:")
    assert first.source_id == second.source_id
