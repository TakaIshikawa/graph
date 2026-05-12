from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "store.db"))
    yield store
    store.close()


def _unit(
    unit_id: str,
    source_project: SourceProject | str,
    tags: list[str],
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=f"Unit {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def test_source_tag_overlap_returns_deterministic_source_pairs(store: Store):
    store.insert_unit(_unit("max-1", SourceProject.MAX, ["solar", "storage"]))
    store.insert_unit(_unit("max-2", SourceProject.MAX, ["solar", "grid"]))
    store.insert_unit(_unit("presence-1", SourceProject.PRESENCE, ["solar", "storage"]))
    store.insert_unit(_unit("presence-2", SourceProject.PRESENCE, ["solar", "finance"]))
    store.insert_unit(_unit("csv-1", SourceProject.CSV, ["solar", "storage", "grid"]))

    rows = store.source_tag_overlap()

    assert rows == [
        {
            "source_a": "csv",
            "source_b": "max",
            "shared_tag_count": 3,
            "shared_unit_count": 3,
            "jaccard": 1.0,
            "top_shared_tags": [
                {"tag": "solar", "source_a_count": 1, "source_b_count": 2},
                {"tag": "grid", "source_a_count": 1, "source_b_count": 1},
                {"tag": "storage", "source_a_count": 1, "source_b_count": 1},
            ],
        },
        {
            "source_a": "csv",
            "source_b": "presence",
            "shared_tag_count": 2,
            "shared_unit_count": 3,
            "jaccard": 0.5,
            "top_shared_tags": [
                {"tag": "solar", "source_a_count": 1, "source_b_count": 2},
                {"tag": "storage", "source_a_count": 1, "source_b_count": 1},
            ],
        },
        {
            "source_a": "max",
            "source_b": "presence",
            "shared_tag_count": 2,
            "shared_unit_count": 4,
            "jaccard": 0.5,
            "top_shared_tags": [
                {"tag": "solar", "source_a_count": 2, "source_b_count": 2},
                {"tag": "storage", "source_a_count": 1, "source_b_count": 1},
            ],
        },
    ]


def test_source_tag_overlap_filters_and_limits(store: Store):
    store.insert_unit(_unit("a", "alpha", ["one", "two", "three"]))
    store.insert_unit(_unit("b", "beta", ["one", "two"]))
    store.insert_unit(_unit("c", "gamma", ["one"]))

    assert [
        (row["source_a"], row["source_b"]) for row in store.source_tag_overlap(min_shared_tags=2)
    ] == [("alpha", "beta")]
    assert len(store.source_tag_overlap(limit=1)) == 1
    assert store.source_tag_overlap(limit=0) == []


def test_source_tag_overlap_empty_or_disjoint_sources_return_empty(store: Store):
    assert store.source_tag_overlap() == []
    store.insert_unit(_unit("a", "alpha", ["one"]))
    store.insert_unit(_unit("b", "beta", ["two"]))
    assert store.source_tag_overlap() == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_shared_tags": 0}, "min_shared_tags must be a positive integer"),
        ({"min_shared_tags": True}, "min_shared_tags must be a positive integer"),
        ({"limit": -1}, "limit must be a non-negative integer or None"),
        ({"limit": True}, "limit must be a non-negative integer or None"),
    ],
)
def test_source_tag_overlap_validates_options(store: Store, kwargs, message):
    with pytest.raises(ValueError, match=message):
        store.source_tag_overlap(**kwargs)
