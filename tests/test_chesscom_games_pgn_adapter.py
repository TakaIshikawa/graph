from __future__ import annotations

from graph.adapters.chesscom_games_pgn import ChesscomGamesPgnAdapter


def test_chesscom_games_pgn_parses_multiple_games_and_retains_tags_and_moves(tmp_path):
    export = tmp_path / "games.pgn"
    export.write_text(
        '[Event "Live Chess"]\n[Site "Chess.com"]\n[UTCDate "2026.05.01"]\n[UTCTime "12:00:00"]\n[White "ada"]\n[Black "grace"]\n[WhiteElo "1500"]\n[BlackElo "1510"]\n[Result "1-0"]\n[TimeControl "600"]\n[ECO "C20"]\n[Opening "King Pawn"]\n[Termination "ada won"]\n[Link "https://www.chess.com/game/live/1"]\n\n1. e4 e5 2. Qh5 Nc6 1-0\n\n'
        '[Event "Live Chess"]\n[UTCDate "2026.05.02"]\n[White "hopper"]\n[Black "lovelace"]\n[Result "0-1"]\n\n1. d4 d5 0-1\n',
        encoding="utf-8",
    )

    units = ChesscomGamesPgnAdapter(path=str(export)).ingest().units

    assert len(units) == 2
    first = units[0]
    assert first.metadata["players"] == {"white": "ada", "black": "grace"}
    assert first.metadata["white_rating"] == "1500"
    assert first.metadata["black_rating"] == "1510"
    assert first.metadata["result"] == "1-0"
    assert first.metadata["time_control"] == "600"
    assert first.metadata["opening"] == "King Pawn"
    assert first.metadata["eco"] == "C20"
    assert first.metadata["termination"] == "ada won"
    assert first.metadata["url"] == "https://www.chess.com/game/live/1"
    assert first.metadata["move_text"] == "1. e4 e5 2. Qh5 Nc6 1-0"
    assert first.metadata["pgn_tags"]["Event"] == "Live Chess"
