from __future__ import annotations

import json

from graph.adapters.discord_threads_json import DiscordThreadsJsonAdapter
from graph.adapters.registry import get_adapter


def test_discord_threads_json_groups_chronologically_and_keeps_media_metadata(tmp_path):
    path = tmp_path / "discord.json"
    path.write_text(json.dumps({"channel": {"id": "C1"}, "messages": [{"id": "2", "thread_id": "T1", "timestamp": "2026-01-02T00:00:00+00:00", "author": {"username": "Bea"}, "content": "Second", "embeds": [{"url": "https://e.test"}]}, {"id": "1", "thread_id": "T1", "timestamp": "2026-01-01T00:00:00+00:00", "author": {"username": "Ada"}, "content": "First", "attachments": [{"url": "https://a.test"}]}]}), encoding="utf-8")

    unit = DiscordThreadsJsonAdapter(str(path)).ingest().units[0]

    assert unit.source_id.startswith("discord_threads_json:")
    assert unit.content.splitlines() == ["Ada: First", "Bea: Second"]
    assert unit.metadata["attachments"] == [{"url": "https://a.test"}]
    assert unit.metadata["embed_urls"] == ["https://e.test"]
    assert isinstance(get_adapter("discord_threads_json"), DiscordThreadsJsonAdapter)
