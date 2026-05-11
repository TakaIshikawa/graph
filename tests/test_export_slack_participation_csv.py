from __future__ import annotations

from datetime import datetime, timezone

from graph.export.slack_participation_csv import export_units_to_slack_participation_csv
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def _unit(source_id: str, *, channel: str = "", user: str = "", ts: str, title: str = "Message", metadata: dict | None = None):
    created_at = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    meta = {"channel": channel, "user": user, "datetime": created_at.isoformat()}
    meta.update(metadata or {})
    return KnowledgeUnit(
        source_project=SourceProject.SLACK_JSON,
        source_id=source_id,
        source_entity_type="slack_message",
        title=title,
        content=title,
        metadata=meta,
        tags=["slack"],
        created_at=created_at,
        updated_at=created_at,
    )


def test_slack_participation_csv_groups_by_channel_and_user_defensively():
    text = export_units_to_slack_participation_csv(
        [
            _unit("2", channel="general", user="U1", ts="2025-01-02T00:00:00Z", metadata={"is_thread_reply": True}),
            _unit("1", channel="general", user="U1", ts="2025-01-01T00:00:00Z", metadata={"reactions": [{"name": "thumbsup", "count": 3}]}),
            _unit("3", ts="2025-01-03T00:00:00Z"),
        ]
    )

    assert text == (
        "channel,user,message_count,first_message_at,last_message_at,thread_reply_count,reaction_count,top_titles\n"
        "general,U1,2,2025-01-01T00:00:00+00:00,2025-01-02T00:00:00+00:00,1,3,Message\n"
        "unknown,unknown,1,2025-01-03T00:00:00+00:00,2025-01-03T00:00:00+00:00,0,0,Message\n"
    )


def test_slack_participation_csv_writes_path(tmp_path):
    path = tmp_path / "slack.csv"
    stats = export_units_to_slack_participation_csv([_unit("1", channel="c", user="u", ts="2025-01-01T00:00:00Z")], path)

    assert stats == {"path": str(path), "rows_written": 1}
    assert path.read_text(encoding="utf-8").startswith("channel,user,message_count")
