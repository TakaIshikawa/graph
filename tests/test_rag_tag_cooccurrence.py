from __future__ import annotations

import pytest

from graph.rag import build_tag_cooccurrence_matrix
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


def unit(unit_id: str, tags: list[str]) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Unit {unit_id}",
        content=f"Content for {unit_id}",
        content_type=ContentType.INSIGHT,
        tags=tags,
    )


def test_build_tag_cooccurrence_matrix_counts_pairs_and_tags():
    units = [
        unit("unit-a", ["Solar", "Storage", "Grid"]),
        unit("unit-b", ["solar", "storage"]),
        unit("unit-c", ["storage", "Finance"]),
    ]

    result = build_tag_cooccurrence_matrix(units)

    assert result["tags"] == [
        {
            "tag": "storage",
            "key": "storage",
            "count": 3,
            "unit_ids": ["unit-a", "unit-b", "unit-c"],
        },
        {
            "tag": "Solar",
            "key": "solar",
            "count": 2,
            "unit_ids": ["unit-a", "unit-b"],
        },
        {
            "tag": "Finance",
            "key": "finance",
            "count": 1,
            "unit_ids": ["unit-c"],
        },
        {
            "tag": "Grid",
            "key": "grid",
            "count": 1,
            "unit_ids": ["unit-a"],
        },
    ]
    assert result["pairs"] == [
        {
            "source": "Solar",
            "target": "storage",
            "source_key": "solar",
            "target_key": "storage",
            "count": 2,
            "unit_ids": ["unit-a", "unit-b"],
        },
        {
            "source": "Finance",
            "target": "storage",
            "source_key": "finance",
            "target_key": "storage",
            "count": 1,
            "unit_ids": ["unit-c"],
        },
        {
            "source": "Grid",
            "target": "Solar",
            "source_key": "grid",
            "target_key": "solar",
            "count": 1,
            "unit_ids": ["unit-a"],
        },
        {
            "source": "Grid",
            "target": "storage",
            "source_key": "grid",
            "target_key": "storage",
            "count": 1,
            "unit_ids": ["unit-a"],
        },
    ]
    assert result["stats"] == {
        "unit_count": 3,
        "tag_count": 4,
        "pair_count": 4,
        "min_count": 1,
        "limit": None,
    }


def test_duplicate_case_variants_in_one_unit_do_not_inflate_counts():
    units = [
        unit("unit-a", ["Solar", "solar", " SOLAR ", "Storage", "storage"]),
        unit("unit-b", ["solar", "storage"]),
    ]

    result = build_tag_cooccurrence_matrix(units)

    assert result["tags"] == [
        {
            "tag": "solar",
            "key": "solar",
            "count": 2,
            "unit_ids": ["unit-a", "unit-b"],
        },
        {
            "tag": "storage",
            "key": "storage",
            "count": 2,
            "unit_ids": ["unit-a", "unit-b"],
        },
    ]
    assert result["pairs"] == [
        {
            "source": "solar",
            "target": "storage",
            "source_key": "solar",
            "target_key": "storage",
            "count": 2,
            "unit_ids": ["unit-a", "unit-b"],
        }
    ]


def test_min_count_filters_pairs_but_keeps_tag_counts():
    units = [
        unit("unit-a", ["alpha", "beta", "gamma"]),
        unit("unit-b", ["alpha", "beta"]),
        unit("unit-c", ["alpha", "gamma"]),
    ]

    result = build_tag_cooccurrence_matrix(units, min_count=2)

    assert result["pairs"] == [
        {
            "source": "alpha",
            "target": "beta",
            "source_key": "alpha",
            "target_key": "beta",
            "count": 2,
            "unit_ids": ["unit-a", "unit-b"],
        },
        {
            "source": "alpha",
            "target": "gamma",
            "source_key": "alpha",
            "target_key": "gamma",
            "count": 2,
            "unit_ids": ["unit-a", "unit-c"],
        },
    ]
    assert [tag["tag"] for tag in result["tags"]] == ["alpha", "beta", "gamma"]
    assert result["stats"]["pair_count"] == 2


def test_limit_applies_after_deterministic_sorting():
    units = [
        unit("unit-a", ["beta", "delta"]),
        unit("unit-b", ["alpha", "gamma"]),
        unit("unit-c", ["alpha", "beta"]),
        unit("unit-d", ["alpha", "beta"]),
    ]

    first = build_tag_cooccurrence_matrix(units, limit=2)
    second = build_tag_cooccurrence_matrix(reversed(units), limit=2)

    assert first == second
    assert first["pairs"] == [
        {
            "source": "alpha",
            "target": "beta",
            "source_key": "alpha",
            "target_key": "beta",
            "count": 2,
            "unit_ids": ["unit-c", "unit-d"],
        },
        {
            "source": "alpha",
            "target": "gamma",
            "source_key": "alpha",
            "target_key": "gamma",
            "count": 1,
            "unit_ids": ["unit-b"],
        },
    ]
    assert first["stats"]["pair_count"] == 2
    assert first["stats"]["limit"] == 2


def test_zero_limit_returns_counts_without_pairs():
    result = build_tag_cooccurrence_matrix([unit("unit-a", ["alpha", "beta"])], limit=0)

    assert result["pairs"] == []
    assert result["stats"]["pair_count"] == 0
    assert result["stats"]["limit"] == 0


@pytest.mark.parametrize("min_count", [0, -1, "2", None, True])
def test_build_tag_cooccurrence_matrix_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        build_tag_cooccurrence_matrix([], min_count=min_count)


@pytest.mark.parametrize("limit", [-1, "2", True])
def test_build_tag_cooccurrence_matrix_validates_limit(limit):
    with pytest.raises(ValueError, match="limit must be a non-negative integer or None"):
        build_tag_cooccurrence_matrix([], limit=limit)
