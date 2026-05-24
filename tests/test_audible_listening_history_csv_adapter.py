from __future__ import annotations

from graph.adapters.audible_listening_history_csv import AudibleListeningHistoryCsvAdapter


def test_audible_listening_history_csv_emits_listening_events_with_stable_identity(tmp_path):
    export = tmp_path / "history.csv"
    export.write_text(
        "Title,Author,Narrator,ASIN,Started At,Finished At,Duration Listened,Marketplace,Device\n"
        '"Example Book","A. Writer; B. Writer","Voice One",B123,2026-05-01T10:00:00Z,2026-05-01T11:30:00Z,"1 hr 30 min",US,iPhone\n',
        encoding="utf-8",
    )

    first = AudibleListeningHistoryCsvAdapter(path=str(export)).ingest().units[0]
    second = AudibleListeningHistoryCsvAdapter(path=str(export)).ingest().units[0]

    assert first.source_entity_type == "listening_event"
    assert first.source_id == second.source_id
    assert first.metadata["asin"] == "B123"
    assert first.metadata["authors"] == ["A. Writer", "B. Writer"]
    assert first.metadata["narrators"] == ["Voice One"]
    assert first.metadata["duration_listened_seconds"] == 5400
    assert first.metadata["marketplace"] == "US"
    assert first.metadata["device"] == "iPhone"
