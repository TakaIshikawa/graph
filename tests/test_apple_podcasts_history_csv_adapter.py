from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.apple_podcasts_history_csv import ApplePodcastsHistoryCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_apple_podcasts_history_csv_ingests_episode_metadata_and_registry(tmp_path):
    export = tmp_path / "podcasts.csv"
    export.write_text(
        "\n".join(
            [
                "Show,Episode Title,Duration,Progress,Completed,Played At,URL",
                "Tech Show,Episode One,01:00:00,00:30:00,yes,2025-01-02T03:04:05Z,https://example.com/ep1",
            ]
        ),
        encoding="utf-8",
    )

    result = ApplePodcastsHistoryCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.APPLE_PODCASTS_HISTORY_CSV
    assert unit.source_entity_type == "podcast_episode"
    assert unit.metadata["show"] == "Tech Show"
    assert unit.metadata["episode"] == "Episode One"
    assert unit.metadata["duration"] == 3600
    assert unit.metadata["progress"] == 1800
    assert unit.metadata["completed"] is True
    assert unit.metadata["played_at"] == "2025-01-02T03:04:05+00:00"
    assert unit.metadata["url"] == "https://example.com/ep1"
    assert unit.metadata["source_file"] == "podcasts.csv"
    assert unit.metadata["row"]["Episode Title"] == "Episode One"
    assert unit.created_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert get_adapter("apple_podcasts_history_csv", path=str(export)).name == "apple_podcasts_history_csv"


def test_apple_podcasts_history_csv_directory_aliases_and_filters(tmp_path):
    (tmp_path / "one.csv").write_text("Podcast,Title,Duration Seconds,Progress Seconds,Finished,Date Played\nShow A,One,3000,120,0,2025-01-01\n", encoding="utf-8")
    (tmp_path / "two.csv").write_text("Show Name,Name,Length,Position,Complete,Listen Date,Episode URL\nShow B,Two,45.5,30,true,2025-01-03,https://example.com/two\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = ApplePodcastsHistoryCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="apple_podcasts_history_csv", source_entity_type="podcast_episode", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["Two"]
    assert first.units[0].metadata["duration"] == 45.5
    assert first.units[0].metadata["completed"] is True
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["show"]).units == []
