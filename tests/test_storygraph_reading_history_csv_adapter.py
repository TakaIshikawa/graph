from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.storygraph_reading_history_csv import StoryGraphReadingHistoryCsvAdapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_storygraph_reading_history_csv_ingests_single_file(tmp_path):
    path = tmp_path / "storygraph.csv"
    _write_csv(
        path,
        [
            {
                "Title": "A Memory Called Empire",
                "Authors": "Arkady Martine",
                "ISBN13": '="9781250186430"',
                "Date Read": "2026-05-01",
                "Read Count": "2",
                "Star Rating": "4.5",
                "Moods": "adventurous, reflective",
                "Tags": "sci-fi, favorites",
                "Pages": "462",
                "Review": "Dense and worthwhile.",
            }
        ],
    )

    result = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.STORYGRAPH_READING_HISTORY_CSV
    assert unit.source_entity_type == "read"
    assert unit.title == "A Memory Called Empire by Arkady Martine"
    assert unit.created_at == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert unit.metadata["isbn13"] == "9781250186430"
    assert unit.metadata["rating"] == 4.5
    assert unit.metadata["read_count"] == 2
    assert unit.metadata["pages"] == 462
    assert unit.metadata["tags"] == ["sci-fi", "favorites"]
    assert unit.metadata["moods"] == ["adventurous", "reflective"]
    assert unit.metadata["row"]["Review"] == "Dense and worthwhile."
    assert unit.metadata["source_file"] == str(path)
    assert "storygraph" in unit.tags


def test_storygraph_reading_history_csv_directory_and_since_filter(tmp_path):
    _write_csv(tmp_path / "old.csv", [{"Title": "Old Book", "Date Read": "2026-04-01"}])
    _write_csv(tmp_path / "new.csv", [{"Title": "New Book", "Date Read": "2026-05-01"}])
    since = SyncState(
        source_project="storygraph_reading_history_csv",
        source_entity_type="read",
        last_sync_at=datetime(2026, 4, 15, tzinfo=timezone.utc),
    )

    result = StoryGraphReadingHistoryCsvAdapter(path=str(tmp_path)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New Book"]
    assert get_adapter("storygraph_reading_history_csv", path=str(tmp_path)).name == "storygraph_reading_history_csv"


def test_storygraph_reading_history_csv_skips_titleless_and_entity_filter(tmp_path):
    path = tmp_path / "storygraph.csv"
    _write_csv(path, [{"Title": "", "Date Read": "2026-05-01"}, {"Title": "Kept", "Date Read": "not a date"}])

    result = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest()

    assert len(result.units) == 1
    assert result.units[0].title == "Kept"
    assert StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest(entity_types=["movie"]).units == []


def test_storygraph_reading_history_csv_source_ids_are_deterministic(tmp_path):
    path = tmp_path / "storygraph.csv"
    _write_csv(path, [{"Title": "Stable", "Authors": "Ada", "Date Read": "2026-05-01", "Read Count": "1"}])

    first = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest().units[0]
    second = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest().units[0]

    assert first.source_id == second.source_id
