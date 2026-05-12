from __future__ import annotations

import pytest

from graph.graph.service import graph_neighborhood_expansion
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _unit(unit_id: str, title: str | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="note",
        title=title or unit_id.upper(),
        content=f"{unit_id} content",
        content_type=ContentType.INSIGHT,
    )


def _edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        source=EdgeSource.MANUAL,
    )


def test_neighborhood_expansion_groups_units_by_depth_with_parent_paths():
    units = [_unit("a"), _unit("b"), _unit("c"), _unit("d")]
    edges = [
        _edge("a-b", "a", "b", EdgeRelation.REFERENCES),
        _edge("b-c", "b", "c", EdgeRelation.BUILDS_ON),
        _edge("d-a", "d", "a", EdgeRelation.CHALLENGES),
    ]

    result = graph_neighborhood_expansion(
        units,
        edges,
        seed_unit_ids=["a"],
        direction="outgoing",
        max_depth=2,
    )

    assert [group["depth"] for group in result["depths"]] == [0, 1, 2]
    assert result["depths"][1]["units"][0] == {
        "unit_id": "b",
        "unit": {"id": "b", "title": "B", "source_project": "max"},
        "depth": 1,
        "parent_unit_id": "a",
        "via_edge_id": "a-b",
        "via_relation": "references",
        "path_unit_ids": ["a", "b"],
        "path_edge_ids": ["a-b"],
        "path_relations": ["references"],
    }
    assert result["depths"][2]["units"][0]["path_unit_ids"] == ["a", "b", "c"]
    assert result["relation_counts"] == {"builds_on": 1, "references": 1}


def test_neighborhood_expansion_respects_direction_filters_and_limit():
    units = [_unit("a"), _unit("b"), _unit("c"), _unit("d")]
    edges = [
        _edge("b-a", "b", "a", EdgeRelation.REFERENCES),
        _edge("c-a", "c", "a", EdgeRelation.BUILDS_ON),
        _edge("a-d", "a", "d", EdgeRelation.REFERENCES),
    ]

    result = graph_neighborhood_expansion(
        units,
        edges,
        seed_unit_ids=["a"],
        direction="incoming",
        max_depth=1,
        relations=["references"],
        limit=1,
    )

    assert result["expanded_count"] == 1
    assert result["depths"][1]["units"][0]["unit_id"] == "b"
    assert result["relation_counts"] == {"references": 1}


def test_neighborhood_expansion_is_deterministic_for_equivalent_inputs():
    units = [_unit("b", "Same"), _unit("a"), _unit("c", "Same")]
    edges = [
        _edge("a-c", "a", "c", EdgeRelation.REFERENCES),
        _edge("a-b", "a", "b", EdgeRelation.REFERENCES),
    ]

    first = graph_neighborhood_expansion(units, edges, seed_unit_ids=["a"])
    second = graph_neighborhood_expansion(list(reversed(units)), list(reversed(edges)), seed_unit_ids=["a"])

    assert first["depths"] == second["depths"]


def test_neighborhood_expansion_validates_inputs():
    with pytest.raises(ValueError, match="direction must be"):
        graph_neighborhood_expansion([], [], seed_unit_ids=["a"], direction="sideways")
    with pytest.raises(ValueError, match="max_depth must be"):
        graph_neighborhood_expansion([], [], seed_unit_ids=["a"], max_depth=-1)
    with pytest.raises(ValueError, match="limit must be"):
        graph_neighborhood_expansion([], [], seed_unit_ids=["a"], limit=-1)
