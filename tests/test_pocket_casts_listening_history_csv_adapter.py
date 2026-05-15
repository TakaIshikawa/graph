from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.pocket_casts_listening_history_csv import PocketCastsListeningHistoryCsvAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_pocket_casts_listening_history_csv_ingests_listen_metadata_and_registry(tmp_path):
    export = tmp_path / "pocket-casts.csv"
    export.write_text(
        "\n".join(
            [
                "podcast_title,episode_title,url,duration,played_at,completed_at,progress,archived,favorite",
                "Tech Show,Episode One,https://example.com/ep1,01:00:00,2025-01-02T03:04:05Z,2025-01-02T03:30:00Z,00:30:00,yes,true",
            ]
        ),
        encoding="utf-8",
    )

    result = PocketCastsListeningHistoryCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.POCKET_CASTS_LISTENING_HISTORY_CSV
    assert unit.source_entity_type == "podcast_listen"
    assert unit.metadata["podcast"] == "Tech Show"
    assert unit.metadata["episode"] == "Episode One"
    assert unit.metadata["url"] == "https://example.com/ep1"
    assert unit.metadata["duration"] == 3600
    assert unit.metadata["progress"] == 1800
    assert unit.metadata["archived"] is True
    assert unit.metadata["favorite"] is True
    assert unit.metadata["source_file"] == "pocket-casts.csv"
    assert unit.metadata["row"]["episode_title"] == "Episode One"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 30, tzinfo=timezone.utc)
    assert get_adapter("pocket_casts_listening_history_csv", path=str(export)).name == "pocket_casts_listening_history_csv"


def test_pocket_casts_listening_history_csv_skips_bad_rows_and_since_filters(tmp_path):
    (tmp_path / "old.csv").write_text("Podcast,Episode,Played At\nShow A,Old,2025-01-01\n", encoding="utf-8")
    (tmp_path / "new.csv").write_text("Podcast,Episode,Duration Seconds,Progress,Favorite,Played At,URL\nShow B,New,45.5,30,0,2025-01-03,https://example.com/new\n,,,\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_bytes(b"\xff\xfe\x00")

    adapter = PocketCastsListeningHistoryCsvAdapter(path=str(tmp_path))
    sync = SyncState(source_project="pocket_casts_listening_history_csv", source_entity_type="podcast_listen", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert first.units[0].metadata["duration"] == 45.5
    assert first.units[0].metadata["favorite"] is False
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["episode"]).units == []
