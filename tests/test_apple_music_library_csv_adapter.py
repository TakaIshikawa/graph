from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.apple_music_library_csv import AppleMusicLibraryCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_apple_music_library_csv_ingests_song_metadata_and_registry(tmp_path):
    export = tmp_path / "library.csv"
    export.write_text(
        "\n".join(
            [
                "Name,Artist,Album,Genre,Play Count,Rating,Last Played,Persistent ID",
                "Song A,Ada,Album A,Electronic,12,80,2025-01-02 03:04:05,ABC123",
            ]
        ),
        encoding="utf-8",
    )

    result = AppleMusicLibraryCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.APPLE_MUSIC_LIBRARY_CSV
    assert unit.source_entity_type == "song"
    assert unit.metadata["artist"] == "Ada"
    assert unit.metadata["album"] == "Album A"
    assert unit.metadata["genre"] == "Electronic"
    assert unit.metadata["play_count"] == 12
    assert unit.metadata["rating"] == 80
    assert unit.metadata["last_played"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["source_file"] == "library.csv"
    assert unit.metadata["row"]["Persistent ID"] == "ABC123"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == unit.created_at
    assert get_adapter("apple_music_library_csv", path=str(export)).name == "apple_music_library_csv"


def test_apple_music_library_csv_ingests_directory_aliases_and_skips_bad_files(tmp_path):
    (tmp_path / "one.csv").write_text("Title,Artist Name,Album Name,Genres,Plays,My Rating,Last Played Date\nOne,Ada,A,Pop,1,5,2025-01-01\n", encoding="utf-8")
    (tmp_path / "two.csv").write_text("Track Name,Track Artist,Album,Genre,Played Count,Stars,Played At\nTwo,Grace,B,Jazz,2,4,2025-01-03T00:00:00Z\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    result = AppleMusicLibraryCsvAdapter(path=str(tmp_path)).ingest()

    assert sorted(unit.title for unit in result.units) == ["One", "Two"]
    assert {unit.metadata["source_file"] for unit in result.units} == {"one.csv", "two.csv"}


def test_apple_music_library_csv_filters_since_and_entity_types_with_deterministic_ids(tmp_path):
    export = tmp_path / "library.csv"
    export.write_text(
        "\n".join(
            [
                "Title,Artist,Album,Last Played",
                "Old,Ada,A,2025-01-01",
                "New,Ada,A,2025-01-03",
            ]
        ),
        encoding="utf-8",
    )

    adapter = AppleMusicLibraryCsvAdapter(path=str(export))
    sync = SyncState(source_project="apple_music_library_csv", source_entity_type="song", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["album"]).units == []
