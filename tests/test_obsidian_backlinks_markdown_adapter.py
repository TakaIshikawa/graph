from datetime import datetime, timezone

from graph.adapters.obsidian_backlinks_markdown import ObsidianBacklinksMarkdownAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_obsidian_backlinks_markdown_extracts_wiki_embeds_and_markdown_links(tmp_path):
    note = tmp_path / "Daily.md"
    note.write_text("# Daily\nSee [[Project Alpha|Alpha]], ![[Diagram]], and [site](https://example.com).\n", encoding="utf-8")

    unit = ObsidianBacklinksMarkdownAdapter(str(tmp_path)).ingest().units[0]

    assert unit.source_entity_type == "backlink_index"
    assert unit.metadata["title"] == "Daily"
    assert unit.metadata["path"] == "Daily.md"
    assert unit.metadata["outgoing_links"] == ["Project Alpha", "Diagram", "https://example.com"]
    assert unit.metadata["unresolved_link_texts"] == ["Project Alpha|Alpha", "Diagram"]
    assert unit.metadata["link_count"] == 3
    assert unit.tags == ["obsidian", "backlinks"]


def test_obsidian_backlinks_markdown_since_entity_filter_and_registry(tmp_path):
    note = tmp_path / "Note.md"
    note.write_text("[[Target]]", encoding="utf-8")
    since = SyncState(source_project="obsidian_backlinks_markdown", source_entity_type="backlink_index", last_sync_at=datetime.now(timezone.utc))

    assert ObsidianBacklinksMarkdownAdapter(str(tmp_path)).ingest(entity_types=["note"]).units == []
    assert ObsidianBacklinksMarkdownAdapter(str(tmp_path)).ingest(since=since).units == []
    assert get_adapter("obsidian_backlinks_markdown", path=str(tmp_path)).name == "obsidian_backlinks_markdown"
