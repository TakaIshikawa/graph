from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.adapters.product_hunt_votes_json import ProductHuntVotesJsonAdapter
from graph.types.enums import ContentType
from graph.types.models import SyncState


def test_product_hunt_votes_json_ingests_vote_metadata(tmp_path):
    export = tmp_path / "votes.json"
    export.write_text(
        json.dumps(
            {
                "votes": [
                    {
                        "id": "vote-123",
                        "product": {
                            "id": "product-456",
                            "name": "Launch Tool",
                            "tagline": "Ship faster",
                            "url": "https://www.producthunt.com/posts/launch-tool",
                            "makers": [{"name": "Ada"}, {"username": "grace"}],
                            "topics": [{"name": "Developer Tools"}, "Productivity"],
                            "featured_at": "2026-05-01T00:00:00Z",
                            "votes_count": "42",
                            "comments_count": 5,
                        },
                        "voted_at": "2026-05-02T03:04:05Z",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = ProductHuntVotesJsonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    unit = result.units[0]
    assert unit.source_project == "product_hunt_votes_json"
    assert unit.source_id == "product_hunt_votes_json:product_vote:vote-123"
    assert unit.source_entity_type == "product_vote"
    assert unit.content_type == ContentType.ARTIFACT
    assert unit.title == "Launch Tool"
    assert "Ship faster" in unit.content
    assert "Makers: Ada, grace" in unit.content
    assert unit.metadata["name"] == "Launch Tool"
    assert unit.metadata["tagline"] == "Ship faster"
    assert unit.metadata["url"] == "https://www.producthunt.com/posts/launch-tool"
    assert unit.metadata["makers"] == ["Ada", "grace"]
    assert unit.metadata["topics"] == ["Developer Tools", "Productivity"]
    assert unit.metadata["votes_count"] == 42
    assert unit.metadata["comments_count"] == 5
    assert unit.metadata["voted_at"] == "2026-05-02T03:04:05+00:00"
    assert unit.metadata["featured_at"] == "2026-05-01T00:00:00+00:00"
    assert unit.metadata["source_file"] == "votes.json"
    assert unit.metadata["record"]["id"] == "vote-123"
    assert unit.tags == ["producthunt", "product_vote", "Developer Tools", "Productivity"]
    assert unit.created_at == datetime(2026, 5, 1, tzinfo=timezone.utc)
    assert unit.updated_at == datetime(2026, 5, 2, 3, 4, 5, tzinfo=timezone.utc)


def test_product_hunt_votes_json_accepts_supported_containers_and_lists(tmp_path):
    cases = [
        ([{"name": "From list", "url": "https://example.com/list", "upvoted_at": "2026-05-01T00:00:00Z"}], "From list"),
        ({"upvotes": [{"product_name": "From upvotes", "product_url": "https://example.com/upvotes", "voted_at": "2026-05-02T00:00:00Z"}]}, "From upvotes"),
        ({"products": [{"name": "From products", "url": "https://example.com/products", "created_at": "2026-05-03T00:00:00Z"}]}, "From products"),
        ({"items": [{"title": "From items", "discussion_url": "https://example.com/items", "voted_at": "2026-05-04T00:00:00Z"}]}, "From items"),
        ({"data": [{"name": "From data", "url": "https://example.com/data", "voted_at": "2026-05-05T00:00:00Z"}]}, "From data"),
    ]

    for index, (payload, expected_title) in enumerate(cases):
        path = tmp_path / f"case-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        result = ProductHuntVotesJsonAdapter(path=str(path)).ingest()

        assert [unit.title for unit in result.units] == [expected_title]


def test_product_hunt_votes_json_directory_skips_bad_files_dedupes_and_sorts(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    bad = tmp_path / "bad.json"
    first.write_text(
        json.dumps(
            [
                {"id": "2", "name": "Second", "url": "https://example.com/second", "voted_at": "2026-05-02T00:00:00Z"},
                {"id": "1", "name": "First", "url": "https://example.com/first", "voted_at": "2026-05-01T00:00:00Z"},
            ]
        ),
        encoding="utf-8",
    )
    second.write_text(json.dumps([{"id": "2", "name": "Second duplicate", "url": "https://example.com/second", "voted_at": "2026-05-03T00:00:00Z"}]), encoding="utf-8")
    bad.write_text("{bad", encoding="utf-8")

    result = ProductHuntVotesJsonAdapter(path=str(tmp_path)).ingest()

    assert [unit.source_id for unit in result.units] == [
        "product_hunt_votes_json:product_vote:1",
        "product_hunt_votes_json:product_vote:2",
    ]
    assert [unit.title for unit in result.units] == ["First", "Second duplicate"]
    assert [(unit.updated_at, unit.source_id) for unit in result.units] == sorted((unit.updated_at, unit.source_id) for unit in result.units)


def test_product_hunt_votes_json_filters_since_and_entity_type(tmp_path):
    export = tmp_path / "votes.json"
    export.write_text(
        json.dumps(
            [
                {"id": "1", "name": "Old", "url": "https://example.com/old", "voted_at": "2026-05-01T00:00:00Z"},
                {"id": "2", "name": "Boundary", "url": "https://example.com/boundary", "voted_at": "2026-05-02T00:00:00Z"},
                {"id": "3", "name": "New", "url": "https://example.com/new", "voted_at": "2026-05-03T00:00:00Z"},
            ]
        ),
        encoding="utf-8",
    )
    since = SyncState(
        source_project="product_hunt_votes_json",
        source_entity_type="product_vote",
        last_sync_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    skipped = ProductHuntVotesJsonAdapter(path=str(export)).ingest(entity_types=["product_bookmark"])
    result = ProductHuntVotesJsonAdapter(path=str(export)).ingest(since=since)

    assert skipped.units == []
    assert [unit.title for unit in result.units] == ["New"]
