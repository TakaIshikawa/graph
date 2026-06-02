from __future__ import annotations

from graph.adapters.registry import get_adapter
from graph.adapters.zoom_recordings_csv import ZoomRecordingsCsvAdapter


def test_zoom_recordings_csv_ingests_recording_rows(tmp_path):
    export = tmp_path / "zoom.csv"
    export.write_text("Meeting ID,UUID,Topic,Host,Start Time,Duration,Recording Type,Share URL,Transcript URL,File Size\n123,u1,Design review,ada@example.com,2026-05-01T10:00:00Z,45 min,shared_screen,https://zoom.us/rec,https://zoom.us/transcript,1024\n", encoding="utf-8")

    unit = ZoomRecordingsCsvAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "recording"
    assert unit.metadata["duration_seconds"] == 2700
    assert unit.metadata["transcript_url"] == "https://zoom.us/transcript"
    assert "Design review" in unit.content


def test_zoom_recordings_csv_allows_missing_recording_url(tmp_path):
    export = tmp_path / "zoom.csv"
    export.write_text("Meeting ID,Topic\n123,Design review\n", encoding="utf-8")

    assert len(ZoomRecordingsCsvAdapter(path=str(export)).ingest().units) == 1


def test_zoom_recordings_csv_is_registered():
    assert isinstance(get_adapter("zoom-recordings-csv"), ZoomRecordingsCsvAdapter)
