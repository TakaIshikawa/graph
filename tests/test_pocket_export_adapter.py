from __future__ import annotations

from datetime import datetime, timezone

from graph.adapters.pocket_export import PocketExportAdapter
from graph.adapters.registry import get_adapter, list_adapters
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_pocket_export_imports_saved_links(tmp_path):
    export = tmp_path / "ril_export.html"
    export.write_text(
        """
        <ul>
          <li><a href="https://example.com/one" time_added="1735689600" tags="Read Later,AI">Example One</a></li>
          <li><a href="https://example.com/two" time_added="1735776000" status="archive" favorite="1">Example Two</a></li>
        </ul>
        """,
        encoding="utf-8",
    )

    result = PocketExportAdapter(path=str(export)).ingest()

    assert len(result.units) == 2
    first = result.units[0]
    assert first.source_project == SourceProject.POCKET_EXPORT
    assert first.source_id.startswith("pocket_export:")
    assert first.title == "Example One"
    assert first.metadata["url"] == "https://example.com/one"
    assert first.metadata["tags"] == ["read later", "ai"]
    assert first.created_at == datetime(2025, 1, 1, tzinfo=timezone.utc)
    second = result.units[1]
    assert second.metadata["archived"] is True
    assert second.metadata["favorite"] is True


def test_pocket_export_skips_malformed_links_and_filters_since(tmp_path):
    export = tmp_path / "ril_export.html"
    export.write_text(
        """
        <a>Missing URL</a>
        <a href="https://old.test" time_added="1735689600">Old</a>
        <a href="https://new.test" time_added="1735862400">New</a>
        """,
        encoding="utf-8",
    )
    since = SyncState(
        source_project="pocket_export",
        source_entity_type="saved_item",
        last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    result = PocketExportAdapter(path=str(export)).ingest(since=since)

    assert [unit.title for unit in result.units] == ["New"]


def test_pocket_export_adapter_is_registered():
    assert "pocket_export" in list_adapters()
    assert get_adapter("pocket_export", path="/tmp/export.html").name == "pocket_export"
