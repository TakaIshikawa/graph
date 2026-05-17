from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.rescuetime_daily_csv import RescueTimeDailyCsvAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_rescuetime_daily_csv_ingests_daily_summary_rows(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "Date,Productivity Pulse,Very Productive,Productive,Neutral,Distracting,Very Distracting,Total Hours,Total Seconds,Category,Activity,Details\n"
        "2026-05-01,78,2.5,3.25,1,0.5,0.25,7.5,27000,Software Development,Code Editor,VS Code\n",
        encoding="utf-8",
    )

    result = RescueTimeDailyCsvAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "rescuetime_daily_csv"
    assert unit.source_entity_type == "daily_activity"
    assert unit.content_type == ContentType.METADATA
    assert unit.metadata["date"] == "2026-05-01"
    assert unit.metadata["productivity_pulse"] == 78
    assert unit.metadata["very_productive"] == 2.5
    assert unit.metadata["productive"] == 3.25
    assert unit.metadata["neutral"] == 1
    assert unit.metadata["distracting"] == 0.5
    assert unit.metadata["very_distracting"] == 0.25
    assert unit.metadata["total_hours"] == 7.5
    assert unit.metadata["total_seconds"] == 27000
    assert unit.metadata["category"] == "Software Development"
    assert unit.metadata["activity"] == "Code Editor"
    assert unit.metadata["details"] == "VS Code"
    assert unit.created_at == datetime(2026, 5, 1, tzinfo=timezone.utc)


def test_rescuetime_daily_csv_skips_blank_rows_and_sorts_stably(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "Date,Productivity Pulse,Total Hours,Category,Activity,Details\n"
        ",,,,,\n"
        "2026-05-02,60,1,Communication,Slack,Messages\n"
        "2026-05-01,90,4,Software Development,Editor,Code\n",
        encoding="utf-8",
    )

    result = RescueTimeDailyCsvAdapter(path=str(export)).ingest()

    assert [(unit.created_at, unit.source_id) for unit in result.units] == sorted((unit.created_at, unit.source_id) for unit in result.units)
    assert [unit.metadata["date"] for unit in result.units] == ["2026-05-01", "2026-05-02"]


def test_rescuetime_daily_csv_filters_since_by_utc_date_timestamp(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text(
        "Date,Productivity Pulse,Total Hours\n"
        "2026-05-01,75,6\n"
        "2026-05-02,80,7\n",
        encoding="utf-8",
    )
    since = SyncState(source_project="rescuetime_daily_csv", source_entity_type="daily_activity", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = RescueTimeDailyCsvAdapter(path=str(export)).ingest(since=since)

    assert [unit.metadata["date"] for unit in result.units] == ["2026-05-02"]


def test_rescuetime_daily_csv_entity_types_filtering(tmp_path):
    export = tmp_path / "rescuetime.csv"
    export.write_text("Date,Productivity Pulse\n2026-05-01,75\n", encoding="utf-8")

    assert RescueTimeDailyCsvAdapter(path=str(export)).ingest(entity_types=["highlight"]).units == []
    assert len(RescueTimeDailyCsvAdapter(path=str(export)).ingest(entity_types=["daily_activity"]).units) == 1
