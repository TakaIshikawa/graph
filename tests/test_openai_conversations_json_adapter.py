import json
from datetime import datetime, timezone

from graph.adapters.openai_conversations_json import OpenAIConversationsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_openai_conversations_json_ingests_mapping_messages(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps({"conversations": [{"id": "conv1", "title": "Adapter help", "create_time": 1770000000, "update_time": 1770000100, "mapping": {"a": {"message": {"author": {"role": "user"}, "content": {"parts": ["Build this"]}, "model": "gpt-5"}}, "b": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Done"]}, "model": "gpt-5"}}}}]}), encoding="utf-8")

    unit = OpenAIConversationsJsonAdapter(str(path)).ingest().units[0]

    assert unit.source_id == "openai_conversations_json:conv1"
    assert unit.metadata["message_count"] == 2
    assert unit.metadata["models"] == ["gpt-5"]
    assert "Build this" in unit.content


def test_openai_conversations_json_since_entity_filter_and_registry(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([{"id": "old", "title": "Old", "updated_at": "2026-04-01T00:00:00Z"}, {"id": "new", "title": "New", "updated_at": "2026-05-02T00:00:00Z", "messages": [{"role": "user", "content": "Hi"}]}]), encoding="utf-8")
    since = SyncState(source_project="openai_conversations_json", source_entity_type="conversation", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = OpenAIConversationsJsonAdapter(str(path)).ingest(since=since, entity_types=["conversation"])

    assert [unit.source_id for unit in result.units] == ["openai_conversations_json:new"]
    assert OpenAIConversationsJsonAdapter(str(path)).ingest(entity_types=["message"]).units == []
    assert get_adapter("openai_conversations_json", path=str(path)).name == "openai_conversations_json"
