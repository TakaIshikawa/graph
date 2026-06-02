import json

from graph.adapters.claude_conversations_json import ClaudeConversationsJsonAdapter
from graph.adapters.registry import get_adapter


def test_claude_conversations_json_ingests_nested_content_blocks(tmp_path):
    path = tmp_path / "claude.json"
    path.write_text(json.dumps({"chats": [{"uuid": "c1", "title": "Claude help", "created_at": "2026-05-01T00:00:00Z", "updated_at": "2026-05-02T00:00:00Z", "account": "acct", "project": "graph", "messages": [{"role": "human", "content": [{"type": "text", "text": "Question"}], "model": "claude-opus"}, {"role": "assistant", "content": "Answer", "model": "claude-opus"}]}]}), encoding="utf-8")

    unit = ClaudeConversationsJsonAdapter(str(path)).ingest().units[0]

    assert unit.source_id == "claude_conversations_json:c1"
    assert unit.metadata["message_count"] == 2
    assert unit.metadata["project"] == "graph"
    assert unit.metadata["models"] == ["claude-opus"]
    assert "Question" in unit.content
    assert get_adapter("claude_conversations_json", path=str(path)).name == "claude_conversations_json"


def test_claude_conversations_json_fallback_and_entity_filter(tmp_path):
    path = tmp_path / "claude.json"
    path.write_text(json.dumps([{"title": "No id", "updated_at": "2026-05-02T00:00:00Z", "messages": [{"role": "user", "content": {"text": "Hi"}}]}, {}]), encoding="utf-8")

    result = ClaudeConversationsJsonAdapter(str(path)).ingest(entity_types=["conversation"])

    assert len(result.units) == 1
    assert result.units[0].source_id.startswith("claude_conversations_json:")
    assert ClaudeConversationsJsonAdapter(str(path)).ingest(entity_types=["message"]).units == []
