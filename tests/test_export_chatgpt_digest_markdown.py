from __future__ import annotations

from datetime import datetime, timezone

from graph.export.chatgpt_digest_markdown import export_units_to_chatgpt_digest_markdown
from graph.types.enums import SourceProject
from graph.types.models import KnowledgeUnit


def _unit(source_id: str, created_at: str, *, conversation_id: str = "", title: str = "Chat", content: str = "Content", tags=None):
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    metadata = {"title": title}
    if conversation_id:
        metadata["conversation_id"] = conversation_id
    return KnowledgeUnit(
        source_project=SourceProject.CHATGPT_JSON,
        source_id=source_id,
        source_entity_type="chatgpt_conversation",
        title=title,
        content=content,
        metadata=metadata,
        tags=tags or ["chatgpt"],
        created_at=dt,
        updated_at=dt,
    )


def test_chatgpt_digest_groups_truncates_and_orders_unsorted_units():
    text = export_units_to_chatgpt_digest_markdown(
        [
            _unit("b", "2025-01-02T00:00:00Z", conversation_id="conv", content="second"),
            _unit("a", "2025-01-01T00:00:00Z", conversation_id="conv", content="x" * 140, tags=["chatgpt", "topic"]),
            _unit("missing", "2025-01-03T00:00:00Z", content="fallback"),
        ],
        max_items_per_conversation=1,
    )

    assert "## Chat\n" in text
    assert "- Conversation: conv\n" in text
    assert "- Date range: 2025-01-01 to 2025-01-02\n" in text
    assert "- Units: 2\n" in text
    assert "xxx..." in text
    assert "- Conversation: missing\n" in text


def test_chatgpt_digest_writes_path(tmp_path):
    path = tmp_path / "digest.md"

    stats = export_units_to_chatgpt_digest_markdown([_unit("a", "2025-01-01T00:00:00Z", conversation_id="conv")], path)

    assert stats == {"path": str(path), "bytes_written": path.stat().st_size}
    assert path.read_text(encoding="utf-8").startswith("# ChatGPT Conversation Digest")
