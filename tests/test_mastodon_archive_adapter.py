from __future__ import annotations

import json

from graph.adapters.mastodon import MastodonAdapter
from graph.types.enums import EdgeRelation, EdgeSource


PUBLIC = "https://www.w3.org/ns/activitystreams#Public"


def test_mastodon_archive_emits_reply_edge(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "orderedItems": [
                    {
                        "type": "Create",
                        "object": {
                            "id": "https://example.social/users/alice/statuses/2",
                            "type": "Note",
                            "content": "<p>Replying</p>",
                            "inReplyTo": "https://remote.social/users/bob/statuses/1",
                            "to": [PUBLIC],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MastodonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == "https://example.social/users/alice/statuses/2"
    assert edge.to_unit_id == "https://remote.social/users/bob/statuses/1"
    assert edge.relation == EdgeRelation.REPLIES_TO
    assert edge.source == EdgeSource.SOURCE
    assert edge.metadata["relation_type"] == "mastodon_reply"


def test_mastodon_archive_emits_boost_edge(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "orderedItems": [
                    {
                        "id": "https://example.social/activities/boost-1",
                        "type": "Announce",
                        "actor": "https://example.social/users/alice",
                        "object": {
                            "id": "https://remote.social/users/bob/statuses/3",
                            "type": "Note",
                            "url": "https://remote.social/@bob/3",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MastodonAdapter(path=str(export)).ingest()

    assert result.units == []
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == "https://example.social/activities/boost-1"
    assert edge.to_unit_id == "https://remote.social/users/bob/statuses/3"
    assert edge.relation == EdgeRelation.REFERENCES
    assert edge.metadata["relation_type"] == "mastodon_boost"


def test_mastodon_archive_emits_mention_edge_without_duplicate_units(tmp_path):
    export = tmp_path / "outbox.json"
    export.write_text(
        json.dumps(
            {
                "orderedItems": [
                    {
                        "type": "Create",
                        "object": {
                            "id": "https://example.social/users/alice/statuses/4",
                            "type": "Note",
                            "content": "<p>Hello <span class='h-card'>@bob</span></p>",
                            "tag": [
                                {
                                    "type": "Mention",
                                    "name": "@bob@remote.social",
                                    "href": "https://remote.social/users/bob",
                                }
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MastodonAdapter(path=str(export)).ingest()

    assert len(result.units) == 1
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.from_unit_id == "https://example.social/users/alice/statuses/4"
    assert edge.to_unit_id == "https://remote.social/users/bob"
    assert edge.relation == EdgeRelation.REFERENCES
    assert edge.metadata["relation_type"] == "mastodon_mention"
