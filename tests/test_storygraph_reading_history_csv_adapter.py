from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.registry import get_adapter
from graph.adapters.storygraph_reading_history_csv import StoryGraphReadingHistoryCsvAdapter
from graph.types.enums import EdgeRelation, EdgeSource, SourceProject
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

    reads = [unit for unit in result.units if unit.source_entity_type == "read"]
    assert len(reads) == 1
    unit = reads[0]
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

    assert [unit.title for unit in result.units if unit.source_entity_type == "read"] == ["New Book"]
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

    first = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest(entity_types=["read"]).units[0]
    second = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest(entity_types=["read"]).units[0]

    assert first.source_id == second.source_id


def test_storygraph_reading_history_csv_adds_page_pace_metadata(tmp_path):
    path = tmp_path / "storygraph.csv"
    _write_csv(
        path,
        [
            {
                "Title": "Fast Book",
                "Authors": "Ada",
                "Date Started": "2026-05-01",
                "Date Finished": "2026-05-07",
                "Date Read": "2026-05-07",
                "Pages": "350",
            }
        ],
    )

    unit = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.metadata["started_at"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["finished_at"] == "2026-05-07T00:00:00+00:00"
    assert unit.metadata["reading_days"] == 7
    assert unit.metadata["pages_per_day"] == 50.0
    assert unit.metadata["completion_bucket"] == "week"


def test_storygraph_reading_history_csv_adds_audio_pace_metadata(tmp_path):
    path = tmp_path / "storygraph.csv"
    _write_csv(
        path,
        [
            {
                "Title": "Audio Book",
                "Started": "2026-05-01",
                "Finished": "2026-05-01",
                "Audiobook Duration": "10 hours",
            }
        ],
    )

    unit = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest().units[0]

    assert unit.metadata["duration_minutes"] == 600
    assert unit.metadata["reading_days"] == 1
    assert unit.metadata["minutes_per_day"] == 600.0
    assert unit.metadata["completion_bucket"] == "same_day"


def test_storygraph_reading_history_csv_pace_buckets_and_invalid_numbers(tmp_path):
    path = tmp_path / "storygraph.csv"
    _write_csv(
        path,
        [
            {
                "Title": "Long Book",
                "Date Started": "2026-01-01",
                "Date Finished": "2026-02-15",
                "Pages": "not numeric",
                "Duration": "unknown",
            },
            {"Title": "Unknown Pace", "Pages": "200"},
        ],
    )

    result = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest()
    by_title = {unit.title: unit for unit in result.units}

    assert by_title["Long Book"].metadata["reading_days"] == 46
    assert by_title["Long Book"].metadata["completion_bucket"] == "long_read"
    assert by_title["Long Book"].metadata["pages"] is None
    assert "pages_per_day" not in by_title["Long Book"].metadata
    assert by_title["Unknown Pace"].metadata["completion_bucket"] == "unknown"
    assert "reading_days" not in by_title["Unknown Pace"].metadata


def test_storygraph_reading_history_csv_emits_author_units_and_edges(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _write_csv(
        first,
        [
            {"Title": "First Book", "Authors": "Ada Lovelace", "Date Read": "2026-05-01"},
            {"Title": "Second Book", "Authors": "ada   lovelace", "Date Read": "2026-05-02"},
        ],
    )
    _write_csv(second, [{"Title": "Other Book", "Authors": "Grace Hopper", "Date Read": "2026-05-03"}])

    result = StoryGraphReadingHistoryCsvAdapter(path=str(tmp_path)).ingest()

    reads = [unit for unit in result.units if unit.source_entity_type == "read"]
    authors = sorted((unit for unit in result.units if unit.source_entity_type == "author"), key=lambda unit: unit.title)
    assert len(reads) == 3
    assert [unit.title for unit in authors] == ["Ada Lovelace", "Grace Hopper"]
    ada = authors[0]
    repeat_ada = next(
        unit
        for unit in StoryGraphReadingHistoryCsvAdapter(path=str(tmp_path)).ingest().units
        if unit.source_entity_type == "author" and unit.title == "Ada Lovelace"
    )
    assert ada.source_id == repeat_ada.source_id
    assert ada.metadata["author"] == "Ada Lovelace"
    assert ada.metadata["book_count"] == 2
    assert ada.metadata["titles"] == ["First Book", "Second Book"]
    assert ada.metadata["read_source_ids"] == sorted(
        unit.source_id for unit in reads if unit.metadata["title"] in {"First Book", "Second Book"}
    )
    assert ada.metadata["source_files"] == [str(first)]
    assert len(result.edges) == 3
    assert {edge.relation for edge in result.edges} == {EdgeRelation.CONTAINS}
    assert {edge.source for edge in result.edges} == {EdgeSource.SOURCE}
    assert {edge.from_unit_id for edge in result.edges} == {author.source_id for author in authors}


def test_storygraph_reading_history_csv_author_entity_filtering(tmp_path):
    path = tmp_path / "storygraph.csv"
    _write_csv(path, [{"Title": "Stable", "Authors": "Ada", "Date Read": "2026-05-01"}])

    read_only = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest(entity_types=["read"])
    author_only = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest(entity_types=["author"])
    both = StoryGraphReadingHistoryCsvAdapter(path=str(path)).ingest(entity_types=["read", "author"])

    assert [unit.source_entity_type for unit in read_only.units] == ["read"]
    assert read_only.edges == []
    assert [unit.source_entity_type for unit in author_only.units] == ["author"]
    assert author_only.edges == []
    assert {unit.source_entity_type for unit in both.units} == {"read", "author"}
    assert len(both.edges) == 1
