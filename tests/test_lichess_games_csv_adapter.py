from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.lichess_games_csv import LichessGamesCsvAdapter
from graph.types.models import SyncState


def test_lichess_games_csv_ingests_games_with_metadata_and_moves(tmp_path):
    export = tmp_path / "lichess.csv"
    export.write_text(
        "Game ID,Date,White,Black,Result,Opening,ECO,Time Control,Rated,URL,Moves\n"
        "abc123,2026-05-01,Alice,Bob,1-0,Sicilian Defense,B20,5+0,true,https://lichess.org/abc123,1. e4 c5 2. Nf3\n"
        ",2026-05-02,Carol,Dave,0-1,,,,false,,\n",
        encoding="utf-8",
    )

    result = LichessGamesCsvAdapter(path=str(export)).ingest()

    assert [unit.source_entity_type for unit in result.units] == ["chess_game", "chess_game"]
    unit = result.units[0]
    assert unit.source_id == "lichess_games_csv:abc123"
    assert unit.metadata["opening"] == "Sicilian Defense"
    assert unit.metadata["eco"] == "B20"
    assert unit.metadata["result"] == "1-0"
    assert unit.metadata["rated"] is True
    assert unit.metadata["time_control"] == "5+0"
    assert "1. e4 c5 2. Nf3" in unit.content


def test_lichess_games_csv_url_ids_ignore_moves_and_filters(tmp_path):
    export = tmp_path / "lichess.csv"
    changed_moves = tmp_path / "lichess-changed.csv"
    export.write_text(
        "Date,White,Black,Result,URL,Moves\n"
        "2026-05-01,Alice,Bob,1-0,https://lichess.org/game,1. e4\n"
        "2026-05-03,Carol,Dave,0-1,,1. d4\n",
        encoding="utf-8",
    )
    changed_moves.write_text(
        "Date,White,Black,Result,URL,Moves\n"
        "2026-05-01,Alice,Bob,1-0,https://lichess.org/game,1. e4 c5\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="lichess_games_csv", source_entity_type="chess_game", last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc))

    original_url_unit = LichessGamesCsvAdapter(path=str(export)).ingest().units[0]
    changed_url_unit = LichessGamesCsvAdapter(path=str(changed_moves)).ingest().units[0]
    filtered = LichessGamesCsvAdapter(path=str(export)).ingest(since=since)

    assert original_url_unit.source_id == changed_url_unit.source_id
    assert [unit.metadata["white"] for unit in filtered.units] == ["Carol"]
    assert LichessGamesCsvAdapter(path=str(export)).ingest(entity_types=["player"]).units == []
