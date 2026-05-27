from __future__ import annotations

from graph.adapters.pocket_casts_queue_csv import PocketCastsQueueCsvAdapter


def test_pocket_casts_queue_csv_ingests_episode_progress(tmp_path):
    export = tmp_path / "queue.csv"
    export.write_text(
        "Podcast Title,Episode Title,Episode URL,Published Date,Added Date,Duration,Playback Position\nThe Show,Episode 1,https://pod.example/1,2024-01-01T00:00:00Z,2024-01-02T00:00:00Z,01:00:00,00:10:00\n",
        encoding="utf-8",
    )

    unit = PocketCastsQueueCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.title == "Episode 1"
    assert unit.metadata["podcast_title"] == "The Show"
    assert unit.metadata["url"] == "https://pod.example/1"
    assert unit.metadata["duration_seconds"] == 3600
    assert unit.metadata["playback_position_seconds"] == 600


def test_pocket_casts_queue_csv_skips_missing_title_and_stable_fallback(tmp_path):
    blank = tmp_path / "blank.csv"
    blank.write_text("Podcast Title,Episode Title\nThe Show,\n", encoding="utf-8")
    fallback = tmp_path / "fallback.csv"
    fallback.write_text("Podcast Title,Episode Title\nThe Show,Episode 1\n", encoding="utf-8")

    assert PocketCastsQueueCsvAdapter(path=str(blank)).ingest().units == []
    first = PocketCastsQueueCsvAdapter(path=str(fallback)).ingest().units[0]
    second = PocketCastsQueueCsvAdapter(path=str(fallback)).ingest().units[0]
    assert first.source_id == second.source_id
