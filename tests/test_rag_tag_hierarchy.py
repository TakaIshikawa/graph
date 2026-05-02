from __future__ import annotations

import pytest

from graph.rag import build_tag_hierarchy
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


def test_build_tag_hierarchy_returns_nested_tag_nodes():
    units = [
        unit("unit-a", ["ai/agents/tools", "ai/agents"]),
        unit("unit-b", ["ai/agents/planning"]),
        unit("unit-c", ["ai/evals"]),
    ]

    assert build_tag_hierarchy(units) == [
        {
            "tag": "ai",
            "parent": None,
            "depth": 0,
            "count": 3,
            "unit_ids": ["unit-a", "unit-b", "unit-c"],
            "children": [
                {
                    "tag": "ai/agents",
                    "parent": "ai",
                    "depth": 1,
                    "count": 2,
                    "unit_ids": ["unit-a", "unit-b"],
                    "children": [
                        {
                            "tag": "ai/agents/planning",
                            "parent": "ai/agents",
                            "depth": 2,
                            "count": 1,
                            "unit_ids": ["unit-b"],
                            "children": [],
                        },
                        {
                            "tag": "ai/agents/tools",
                            "parent": "ai/agents",
                            "depth": 2,
                            "count": 1,
                            "unit_ids": ["unit-a"],
                            "children": [],
                        },
                    ],
                },
                {
                    "tag": "ai/evals",
                    "parent": "ai",
                    "depth": 1,
                    "count": 1,
                    "unit_ids": ["unit-c"],
                    "children": [],
                },
            ],
        }
    ]


def test_build_tag_hierarchy_preserves_flat_tags_as_sorted_roots():
    units = [
        unit("unit-a", ["research", "ai/agents"]),
        unit("unit-b", ["research", "archive"]),
        unit("unit-c", ["ai/evals"]),
    ]

    hierarchy = build_tag_hierarchy(units)

    assert [node["tag"] for node in hierarchy] == ["ai", "research", "archive"]
    assert hierarchy[0]["children"] == [
        {
            "tag": "ai/agents",
            "parent": "ai",
            "depth": 1,
            "count": 1,
            "unit_ids": ["unit-a"],
            "children": [],
        },
        {
            "tag": "ai/evals",
            "parent": "ai",
            "depth": 1,
            "count": 1,
            "unit_ids": ["unit-c"],
            "children": [],
        },
    ]


def test_build_tag_hierarchy_min_count_prunes_low_frequency_nodes():
    units = [
        unit("unit-a", ["ai/agents/tools", "research"]),
        unit("unit-b", ["ai/agents/planning", "research"]),
        unit("unit-c", ["ai/evals"]),
    ]

    assert build_tag_hierarchy(units, min_count=2) == [
        {
            "tag": "ai",
            "parent": None,
            "depth": 0,
            "count": 3,
            "unit_ids": ["unit-a", "unit-b", "unit-c"],
            "children": [
                {
                    "tag": "ai/agents",
                    "parent": "ai",
                    "depth": 1,
                    "count": 2,
                    "unit_ids": ["unit-a", "unit-b"],
                    "children": [],
                }
            ],
        },
        {
            "tag": "research",
            "parent": None,
            "depth": 0,
            "count": 2,
            "unit_ids": ["unit-a", "unit-b"],
            "children": [],
        },
    ]


def test_build_tag_hierarchy_counts_repeated_tags_once_per_unit():
    units = [
        unit("unit-a", ["ai/agents", "ai/agents", "ai/agents/tools"]),
        unit("unit-b", ["ai/agents"]),
    ]

    hierarchy = build_tag_hierarchy(units)

    assert hierarchy[0]["count"] == 2
    assert hierarchy[0]["unit_ids"] == ["unit-a", "unit-b"]
    assert hierarchy[0]["children"][0]["count"] == 2
    assert hierarchy[0]["children"][0]["unit_ids"] == ["unit-a", "unit-b"]
    assert hierarchy[0]["children"][0]["children"][0]["count"] == 1
    assert hierarchy[0]["children"][0]["children"][0]["unit_ids"] == ["unit-a"]


def test_build_tag_hierarchy_empty_units_returns_empty_list():
    assert build_tag_hierarchy([]) == []


@pytest.mark.parametrize("min_count", [0, -1, "2", None, True])
def test_build_tag_hierarchy_validates_min_count(min_count):
    with pytest.raises(ValueError, match="min_count must be a positive integer"):
        build_tag_hierarchy([], min_count=min_count)
