from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.product_hunt_bookmarks_json import ProductHuntBookmarksJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.enums import SourceProject
from graph.types.models import SyncState


def test_product_hunt_bookmarks_json_ingests_nested_metadata_and_registry(tmp_path):
    export = tmp_path / "producthunt.json"
    export.write_text(
        json.dumps(
            {
                "products": [
                    {
                        "name": "Launch Tool",
                        "tagline": "Ship faster",
                        "description": "A useful product",
                        "url": "https://www.producthunt.com/posts/launch-tool",
                        "makers": [{"name": "Ada"}, "Grace"],
                        "topics": [{"name": "Developer Tools"}, "Productivity"],
                        "votes": "42",
                        "comments_count": 5,
                        "saved_at": "2025-01-02T03:04:05Z",
                        "featured_at": "2025-01-01T00:00:00Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ProductHuntBookmarksJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == SourceProject.PRODUCT_HUNT_BOOKMARKS_JSON
    assert unit.source_entity_type == "product_bookmark"
    assert unit.metadata["name"] == "Launch Tool"
    assert unit.metadata["tagline"] == "Ship faster"
    assert unit.metadata["url"] == "https://www.producthunt.com/posts/launch-tool"
    assert unit.metadata["makers"] == ["Ada", "Grace"]
    assert unit.metadata["topics"] == ["Developer Tools", "Productivity"]
    assert unit.metadata["votes"] == 42
    assert unit.metadata["source_file"] == "producthunt.json"
    assert unit.updated_at == datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert get_adapter("product_hunt_bookmarks_json", path=str(export)).name == "product_hunt_bookmarks_json"


def test_product_hunt_bookmarks_json_skips_bad_records_files_since_and_filters(tmp_path):
    (tmp_path / "old.json").write_text(json.dumps([{"name": "Old", "url": "https://example.com/old", "saved_at": "2025-01-01T00:00:00Z"}]), encoding="utf-8")
    (tmp_path / "new.json").write_text(json.dumps({"items": [{"name": "New", "url": "https://example.com/new", "saved_at": "2025-01-03T00:00:00Z"}, {"tagline": "No identity"}]}), encoding="utf-8")
    (tmp_path / "bad.json").write_text("{bad", encoding="utf-8")

    adapter = ProductHuntBookmarksJsonAdapter(path=str(tmp_path))
    sync = SyncState(source_project="product_hunt_bookmarks_json", source_entity_type="product_bookmark", last_sync_at=datetime(2025, 1, 2, tzinfo=timezone.utc))
    first = adapter.ingest(since=sync)
    second = adapter.ingest(since=sync)

    assert [unit.title for unit in first.units] == ["New"]
    assert [unit.source_id for unit in first.units] == [unit.source_id for unit in second.units]
    assert adapter.ingest(entity_types=["post"]).units == []
