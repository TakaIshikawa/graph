from __future__ import annotations

from datetime import datetime, timezone

from graph.export.activitywatch_focus_sessions_csv import export_units_to_activitywatch_focus_sessions_csv
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def _unit(source_id: str, timestamp: str, duration: float, app: str, title: str, domain: str = ""):
    return KnowledgeUnit(
        source_project=SourceProject.ACTIVITYWATCH_JSON,
        source_id=source_id,
        source_entity_type="activity",
        title=title,
        content=title,
        content_type=ContentType.METADATA,
        tags=["activitywatch"],
        metadata={"timestamp": timestamp, "duration": duration, "app": app, "title": title, "domain": domain},
        created_at=datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
        updated_at=datetime.fromisoformat(timestamp.replace("Z", "+00:00")),
    )


def test_activitywatch_focus_sessions_csv_merges_adjacent_units_and_filters_short_sessions():
    text = export_units_to_activitywatch_focus_sessions_csv(
        [
            _unit("c", "2025-01-01T10:20:00Z", 60, "Code", "Other"),
            _unit("a", "2025-01-01T10:00:00Z", 180, "Code", "main.py"),
            _unit("b", "2025-01-01T10:03:00Z", 180, "Code", "main.py"),
        ],
        min_duration_seconds=300,
    )

    assert text == (
        "session_start,session_end,duration_seconds,app,domain,title,unit_count\n"
        "2025-01-01T10:00:00+00:00,2025-01-01T10:06:00+00:00,360,Code,,main.py,2\n"
    )


def test_activitywatch_focus_sessions_csv_writes_path(tmp_path):
    path = tmp_path / "focus.csv"

    stats = export_units_to_activitywatch_focus_sessions_csv(
        [_unit("a", "2025-01-01T10:00:00Z", 300, "Firefox", "Docs", "example.com")],
        path,
    )

    assert stats == {"path": str(path), "rows_written": 1}
    assert path.read_text(encoding="utf-8").splitlines()[1].endswith(",Firefox,example.com,Docs,1")
