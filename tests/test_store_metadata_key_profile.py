from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    yield store
    store.close()


def unit(source_id: str, metadata: dict) -> KnowledgeUnit:
    return KnowledgeUnit(
        source_project=SourceProject.MAX,
        source_id=source_id,
        source_entity_type="insight",
        title=f"Unit {source_id}",
        content=f"Content {source_id}",
        content_type=ContentType.INSIGHT,
        metadata=metadata,
    )


def test_metadata_key_profile_counts_distinct_values_types_and_samples(store: Store):
    store.insert_unit(
        unit(
            "a",
            {
                "review": {"state": "approved", "score": 1, "active": True},
                "owner": "alice",
            },
        )
    )
    store.insert_unit(
        unit(
            "b",
            {
                "review": {"state": "approved", "score": 1.5, "active": False},
                "owner": "bob",
            },
        )
    )
    store.insert_unit(
        unit(
            "c",
            {
                "review": {"state": "draft", "score": "high", "active": True},
                "owner": "alice",
            },
        )
    )
    store.insert_unit(unit("d", {}))

    rows = store.metadata_key_profile()
    by_key = {row["key"]: row for row in rows}

    assert [row["key"] for row in rows] == [
        "owner",
        "review.active",
        "review.score",
        "review.state",
    ]
    assert by_key["review.state"] == {
        "key": "review.state",
        "occurrence_count": 3,
        "distinct_value_count": 2,
        "value_types": ["string"],
        "sample_values": ["approved", "draft"],
    }
    assert by_key["owner"]["occurrence_count"] == 3
    assert by_key["owner"]["distinct_value_count"] == 2
    assert by_key["owner"]["sample_values"] == ["alice", "bob"]
    assert by_key["review.score"] == {
        "key": "review.score",
        "occurrence_count": 3,
        "distinct_value_count": 3,
        "value_types": ["integer", "number", "string"],
        "sample_values": [1, 1.5, "high"],
    }
    assert by_key["review.active"] == {
        "key": "review.active",
        "occurrence_count": 3,
        "distinct_value_count": 2,
        "value_types": ["boolean"],
        "sample_values": [False, True],
    }


def test_metadata_key_profile_keeps_mixed_scalar_types_distinct(store: Store):
    store.insert_unit(unit("a", {"flag": True, "rank": 1, "note": None}))
    store.insert_unit(unit("b", {"flag": 1, "rank": 1.0, "note": None}))
    store.insert_unit(unit("c", {"flag": "1", "rank": "1", "note": "none"}))

    by_key = {row["key"]: row for row in store.metadata_key_profile()}

    assert by_key["flag"] == {
        "key": "flag",
        "occurrence_count": 3,
        "distinct_value_count": 3,
        "value_types": ["boolean", "integer", "string"],
        "sample_values": [True, 1, "1"],
    }
    assert by_key["rank"] == {
        "key": "rank",
        "occurrence_count": 3,
        "distinct_value_count": 3,
        "value_types": ["integer", "number", "string"],
        "sample_values": [1, 1.0, "1"],
    }
    assert by_key["note"] == {
        "key": "note",
        "occurrence_count": 3,
        "distinct_value_count": 2,
        "value_types": ["null", "string"],
        "sample_values": [None, "none"],
    }


def test_metadata_key_profile_filters_limits_and_caps_samples(store: Store):
    store.insert_unit(unit("a", {"review": {"state": "approved", "owner": "alice"}}))
    store.insert_unit(unit("b", {"review": {"state": "draft", "owner": "bob"}}))
    store.insert_unit(unit("c", {"review": {"state": "needs-work", "owner": "carol"}}))
    store.insert_unit(unit("d", {"other": "ignored"}))

    assert store.metadata_key_profile(prefix="review", limit=1, sample_size=2) == [
        {
            "key": "review.owner",
            "occurrence_count": 3,
            "distinct_value_count": 3,
            "value_types": ["string"],
            "sample_values": ["alice", "bob"],
        }
    ]


def test_metadata_key_profile_ignores_units_without_metadata(store: Store):
    store.insert_unit(unit("a", {}))
    store.insert_unit(unit("b", {"topic": "storage"}))

    assert store.metadata_key_profile() == [
        {
            "key": "topic",
            "occurrence_count": 1,
            "distinct_value_count": 1,
            "value_types": ["string"],
            "sample_values": ["storage"],
        }
    ]


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_metadata_key_profile_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        store.metadata_key_profile(limit=limit)


@pytest.mark.parametrize("sample_size", [0, -1, 1.5, True])
def test_metadata_key_profile_validates_sample_size(store: Store, sample_size):
    with pytest.raises(ValueError, match="sample_size must be a positive integer"):
        store.metadata_key_profile(sample_size=sample_size)
