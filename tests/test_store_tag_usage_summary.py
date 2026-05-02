from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "graph.db"
    s = Store(str(path))
    yield s
    s.close()
    for candidate in (
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    ):
        candidate.unlink(missing_ok=True)


def unit(
    unit_id: str,
    title: str,
    *,
    tags: list[str],
    source_project: SourceProject | str = SourceProject.MAX,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def test_tag_usage_summary_aggregates_counts_percentages_and_exact_variants(store: Store):
    store.insert_unit(unit("gamma", "Gamma", tags=["solar", "storage"]))
    store.insert_unit(
        unit("alpha", "Alpha", tags=["solar", "storage", "AI"], source_project="custom")
    )
    store.insert_unit(unit("beta", "Beta", tags=["solar", "ai"], source_project=SourceProject.CSV))

    assert store.tag_usage_summary(include_examples=False) == {
        "total_distinct_tags": 4,
        "total_tag_assignments": 7,
        "tags": [
            {"tag": "solar", "count": 3, "percentage": 42.86},
            {"tag": "storage", "count": 2, "percentage": 28.57},
            {"tag": "AI", "count": 1, "percentage": 14.29},
            {"tag": "ai", "count": 1, "percentage": 14.29},
        ],
    }


def test_tag_usage_summary_includes_compact_examples_when_requested(store: Store):
    store.insert_unit(unit("gamma", "Gamma", tags=["solar"]))
    store.insert_unit(unit("alpha", "Alpha", tags=["solar"], source_project="custom"))
    store.insert_unit(unit("beta", "Beta", tags=["solar"], source_project=SourceProject.CSV))
    store.insert_unit(unit("delta", "Delta", tags=["solar"]))

    summary = store.tag_usage_summary()

    assert summary["tags"] == [
        {
            "tag": "solar",
            "count": 4,
            "percentage": 100.0,
            "examples": [
                {
                    "id": "alpha",
                    "title": "Alpha",
                    "source": {
                        "project": "custom",
                        "id": "source-alpha",
                        "entity_type": "note",
                    },
                },
                {
                    "id": "beta",
                    "title": "Beta",
                    "source": {
                        "project": "csv",
                        "id": "source-beta",
                        "entity_type": "note",
                    },
                },
                {
                    "id": "delta",
                    "title": "Delta",
                    "source": {
                        "project": "max",
                        "id": "source-delta",
                        "entity_type": "note",
                    },
                },
            ],
        }
    ]


def test_tag_usage_summary_limit_keeps_totals_and_limits_returned_tags(store: Store):
    store.insert_unit(unit("alpha", "Alpha", tags=["zeta", "beta", "alpha"]))
    store.insert_unit(unit("beta", "Beta", tags=["zeta", "beta"]))
    store.insert_unit(unit("gamma", "Gamma", tags=["zeta"]))

    limited = store.tag_usage_summary(limit=2, include_examples=False)
    empty_limit = store.tag_usage_summary(limit=0)

    assert limited == {
        "total_distinct_tags": 3,
        "total_tag_assignments": 6,
        "tags": [
            {"tag": "zeta", "count": 3, "percentage": 50.0},
            {"tag": "beta", "count": 2, "percentage": 33.33},
        ],
    }
    assert empty_limit == {
        "total_distinct_tags": 3,
        "total_tag_assignments": 6,
        "tags": [],
    }


def test_tag_usage_summary_empty_store_returns_zero_totals(store: Store):
    assert store.tag_usage_summary() == {
        "total_distinct_tags": 0,
        "total_tag_assignments": 0,
        "tags": [],
    }


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_tag_usage_summary_validates_limit(store: Store, limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer"):
        store.tag_usage_summary(limit=limit)
