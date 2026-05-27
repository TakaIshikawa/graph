from __future__ import annotations

import json

from graph.adapters.registry import get_adapter
from graph.adapters.slack_threads_json import SlackThreadsJsonAdapter


def test_slack_threads_json_groups_thread_messages_and_preserves_files(tmp_path):
    path = tmp_path / "C1.json"
    path.write_text(json.dumps([{"ts": "1700000000.000001", "thread_ts": "1700000000.000001", "user": "U1", "text": "Root", "files": [{"id": "F1"}]}, {"ts": "1700000001.000001", "thread_ts": "1700000000.000001", "user": "U2", "text": "Reply"}, {"ts": "1700000002.000001", "user": "U3", "text": "Standalone"}]), encoding="utf-8")

    units = SlackThreadsJsonAdapter(str(path)).ingest().units

    assert len(units) == 2
    thread = next(unit for unit in units if unit.metadata["reply_count"] == 1)
    assert "U1: Root" in thread.content
    assert "U2: Reply" in thread.content
    assert thread.metadata["attachments"] == [[{"id": "F1"}]]
    assert isinstance(get_adapter("slack_threads_json"), SlackThreadsJsonAdapter)
