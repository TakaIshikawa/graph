from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.dropbox_paper_docs_json import DropboxPaperDocsJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_dropbox_paper_docs_json_ingests_documents(tmp_path):
    export = tmp_path / "paper.json"
    export.write_text(
        '{"documents":[{"id":"d1","title":"Launch notes","markdown":"Ship plan","owner":{"email":"ada@example.com"},"created_at":"2026-05-01T10:00:00Z","updated_at":"2026-05-02T10:00:00Z","sharing_url":"https://paper.dropbox.com/doc/d1","path":"/Team","tags":["launch"]}]}',
        encoding="utf-8",
    )

    unit = DropboxPaperDocsJsonAdapter(path=str(export)).ingest().units[0]

    assert unit.source_entity_type == "document"
    assert unit.source_id == "dropbox_paper_docs_json:d1"
    assert unit.title == "Launch notes"
    assert "Ship plan" in unit.content
    assert unit.metadata["owner"] == "ada@example.com"
    assert unit.metadata["source_url"] == "https://paper.dropbox.com/doc/d1"
    assert unit.metadata["folder"] == "/Team"
    assert "launch" in unit.tags


def test_dropbox_paper_docs_json_skips_empty_and_filters(tmp_path):
    export = tmp_path / "paper.json"
    export.write_text('[{},{"title":"Old","text":"Body","updated_at":"2026-01-01T00:00:00Z"}]', encoding="utf-8")

    adapter = DropboxPaperDocsJsonAdapter(path=str(export))

    assert adapter.ingest(entity_types=["task"]).units == []
    assert adapter.ingest(since=SyncState(source_project="x", source_entity_type="document", last_sync_at=datetime(2026, 2, 1, tzinfo=timezone.utc))).units == []


def test_dropbox_paper_docs_json_is_registered():
    assert isinstance(get_adapter("dropbox-paper-docs-json"), DropboxPaperDocsJsonAdapter)
    assert isinstance(get_adapter("dropbox_paper_docs_json"), DropboxPaperDocsJsonAdapter)
