import json
from datetime import datetime, timezone

from graph.adapters import MastodonFavouritesJsonAdapter
from graph.adapters.registry import get_adapter
from graph.types.models import SyncState


def test_mastodon_favourites_json_ingests_wrapped_statuses(tmp_path):
    path = tmp_path / "favourites.json"
    path.write_text(
        json.dumps(
            {
                "favourites": [
                    {
                        "id": "fav-1",
                        "status": {
                            "id": "101",
                            "url": "https://m.example/@ada/101",
                            "account": {"display_name": "Ada", "acct": "ada@example"},
                            "created_at": "2026-05-01T10:00:00Z",
                            "content": "<p>Hello <strong>world</strong></p>",
                            "tags": [{"name": "ai"}],
                            "language": "en",
                            "reblog": {"id": "99", "url": "https://m.example/@grace/99", "account": {"acct": "grace@example"}},
                            "reblogs_count": 2,
                            "favourites_count": 3,
                            "replies_count": 4,
                        },
                    },
                    {"status": {}},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MastodonFavouritesJsonAdapter(str(path)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_entity_type == "favourite"
    assert unit.source_id == "mastodon_favourites_json:101"
    assert unit.title.startswith("Ada: Hello world")
    assert unit.content == "Author: Ada\nHello world\nURL: https://m.example/@ada/101\nReblog: https://m.example/@grace/99"
    assert unit.metadata["status_id"] == "101"
    assert unit.metadata["reblog_status_id"] == "99"
    assert unit.metadata["favourites_count"] == 3
    assert unit.tags == ["mastodon", "favourite", "ai"]


def test_mastodon_favourites_json_flat_records_filters_and_registry(tmp_path):
    path = tmp_path / "flat.json"
    path.write_text(
        json.dumps(
            [
                {"id": "old", "content": "Old", "created_at": "2026-04-01T00:00:00Z"},
                {"url": "https://m.example/no-id", "content": "<p>No id</p>", "created_at": "2026-05-02T00:00:00Z"},
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(source_project="mastodon_favourites_json", source_entity_type="favourite", last_sync_at=datetime(2026, 5, 1, tzinfo=timezone.utc))

    result = MastodonFavouritesJsonAdapter(str(path)).ingest(since=since, entity_types=["favourite"])

    assert len(result.units) == 1
    assert result.units[0].source_id.startswith("mastodon_favourites_json:")
    assert result.units[0].source_id != "mastodon_favourites_json:"
    assert result.units[0].content == "No id\nURL: https://m.example/no-id"
    assert MastodonFavouritesJsonAdapter(str(path)).ingest(entity_types=["unknown"]).units == []
    assert get_adapter("mastodon_favourites_json", path=str(path)).name == "mastodon_favourites_json"


def test_mastodon_favourites_json_handles_alternate_wrappers_and_malformed_json(tmp_path):
    valid = tmp_path / "valid.json"
    invalid = tmp_path / "invalid.json"
    valid.write_text(json.dumps({"orderedItems": [{"object": {"id": "42", "content": "Wrapped"}}]}), encoding="utf-8")
    invalid.write_text("{", encoding="utf-8")

    result = MastodonFavouritesJsonAdapter(str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == ["mastodon_favourites_json:42"]
