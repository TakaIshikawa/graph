from __future__ import annotations

from graph.store.collection_tag_transition_summary import collection_tag_transition_summary
from graph.types.models import KnowledgeUnit


def _unit(unit_id: str, collection: str, tags: list[str], snapshot_at: str):
    return KnowledgeUnit(
        id=unit_id,
        source_project="snapshot",
        source_id=unit_id,
        source_entity_type="member",
        title=unit_id,
        content="",
        tags=tags,
        metadata={"collection": collection, "snapshot_at": snapshot_at},
    )


def test_collection_tag_transition_summary_reports_added_removed_retained_and_empty_sides():
    rows = collection_tag_transition_summary(
        [
            _unit("a-before", "alpha", ["red", "blue"], "2026-05-01T00:00:00+00:00"),
            _unit("a-after", "alpha", ["blue", "green"], "2026-05-02T00:00:00+00:00"),
            _unit("before-only", "archive", ["old"], "2026-05-01T00:00:00+00:00"),
            _unit("after-only", "new", ["new"], "2026-05-02T00:00:00+00:00"),
        ],
        before_at="2026-05-01T12:00:00+00:00",
        after_at="2026-05-02T12:00:00+00:00",
    )

    assert rows == [
        {
            "collection": "alpha",
            "added_count": 1,
            "removed_count": 1,
            "retained_count": 1,
            "net_change": 0,
            "added_tags": ["green"],
            "removed_tags": ["red"],
            "retained_tags": ["blue"],
        },
        {
            "collection": "archive",
            "added_count": 0,
            "removed_count": 1,
            "retained_count": 0,
            "net_change": -1,
            "added_tags": [],
            "removed_tags": ["old"],
            "retained_tags": [],
        },
        {
            "collection": "new",
            "added_count": 1,
            "removed_count": 0,
            "retained_count": 0,
            "net_change": 1,
            "added_tags": ["new"],
            "removed_tags": [],
            "retained_tags": [],
        },
    ]
