from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.adapters.spotify_saved_episodes_csv import SpotifySavedEpisodesCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def _write_csv(path, rows):
    fields = list({key: None for row in rows for key in row.keys()}.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_spotify_saved_episodes_csv_ingests_episode_metadata(tmp_path):
    export = tmp_path / "spotify-episodes.csv"
    _write_csv(
        export,
        [
            {
                "Show Name": "Tech Show",
                "Episode Name": "Episode One",
                "Episode URI": "spotify:episode:abc123",
                "Episode URL": "https://open.spotify.com/episode/abc123",
                "Added At": "2025-01-02T03:04:05Z",
                "Release Date": "2024-12-31",
                "Duration Ms": "3600500",
                "Description": "A useful episode.",
                "Explicit": "false",
            }
        ],
    )

    result = SpotifySavedEpisodesCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "spotify_saved_episodes_csv"
    assert unit.source_id.startswith("spotify_saved_episodes_csv:")
    assert unit.source_entity_type == "podcast_episode"
    assert unit.title == "Episode One"
    assert unit.content_type == ContentType.ARTIFACT
    assert "Episode: Episode One" in unit.content
    assert "Show: Tech Show" in unit.content
    assert "Spotify URI: spotify:episode:abc123" in unit.content
    assert "URL: https://open.spotify.com/episode/abc123" in unit.content
    assert "Added: 2025-01-02T03:04:05+00:00" in unit.content
    assert "Released: 2024-12-31T00:00:00+00:00" in unit.content
    assert "Duration ms: 3600500" in unit.content
    assert "Explicit: False" in unit.content
    assert "A useful episode." in unit.content
    assert unit.metadata["show"] == "Tech Show"
    assert unit.metadata["episode"] == "Episode One"
    assert unit.metadata["episode_uri"] == "spotify:episode:abc123"
    assert unit.metadata["episode_url"] == "https://open.spotify.com/episode/abc123"
    assert unit.metadata["added_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["release_date"] == "2024-12-31T00:00:00+00:00"
    assert unit.metadata["duration_ms"] == 3600500
    assert unit.metadata["description"] == "A useful episode."
    assert unit.metadata["explicit"] is False
    assert unit.metadata["source_file"] == "spotify-episodes.csv"
    assert unit.metadata["row"]["Episode Name"] == "Episode One"
    assert unit.tags == ["spotify", "podcast", "podcast_episode", "Tech Show"]
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_spotify_saved_episodes_csv_directory_filters_and_distinct_duplicate_names(tmp_path):
    _write_csv(
        tmp_path / "old.csv",
        [
            {
                "Show Name": "Show A",
                "Episode Name": "Trailer",
                "Added At": "2025-01-01",
            },
            {},
        ],
    )
    _write_csv(
        tmp_path / "new.csv",
        [
            {
                "Show Name": "Show B",
                "Episode Name": "Trailer",
                "Added At": "2025-01-03",
                "Explicit": "yes",
            },
            {
                "Show Name": "Show C",
                "Episode Name": "Trailer",
                "Added At": "2025-01-04",
                "Duration Ms": "45.5",
            },
            {"Show Name": "", "Episode Name": "   ", "Added At": ""},
        ],
    )
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = SpotifySavedEpisodesCsvAdapter(path=str(tmp_path))
    sync = SyncState(
        source_project="spotify_saved_episodes_csv",
        source_entity_type="podcast_episode",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    units_by_show = {unit.metadata["show"]: unit for unit in first.units}
    assert sorted(units_by_show) == ["Show B", "Show C"]
    assert len({unit.source_id for unit in first.units}) == 2
    assert units_by_show["Show B"].metadata["explicit"] is True
    assert units_by_show["Show C"].metadata["duration_ms"] == 45
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["show"]).units == []
