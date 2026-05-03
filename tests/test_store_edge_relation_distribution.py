from __future__ import annotations

import pytest

from graph.store.db import Store
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


@pytest.fixture
def store(tmp_path):
    store = Store(str(tmp_path / "graph.db"))
    yield store
    store.close()


def _unit(unit_id: str, title: str) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=f"Content for {title}",
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


def test_edge_relation_distribution_empty_graph_returns_empty_dict(store: Store):
    result = store.edge_relation_distribution()

    assert result == {}


def test_edge_relation_distribution_graph_with_no_edges(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha"))
    store.insert_unit(_unit("unit-2", "Beta"))

    result = store.edge_relation_distribution()

    assert result == {}


def test_edge_relation_distribution_single_relation_type(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha"))
    store.insert_unit(_unit("unit-2", "Beta"))
    store.insert_unit(_unit("unit-3", "Gamma"))

    store.insert_edge(_edge("edge-1", "unit-1", "unit-2", EdgeRelation.BUILDS_ON))
    store.insert_edge(_edge("edge-2", "unit-2", "unit-3", EdgeRelation.BUILDS_ON))

    result = store.edge_relation_distribution()

    assert result == {EdgeRelation.BUILDS_ON: 2}


def test_edge_relation_distribution_multiple_relation_types(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha"))
    store.insert_unit(_unit("unit-2", "Beta"))
    store.insert_unit(_unit("unit-3", "Gamma"))
    store.insert_unit(_unit("unit-4", "Delta"))

    store.insert_edge(_edge("edge-1", "unit-1", "unit-2", EdgeRelation.BUILDS_ON))
    store.insert_edge(_edge("edge-2", "unit-2", "unit-3", EdgeRelation.BUILDS_ON))
    store.insert_edge(_edge("edge-3", "unit-1", "unit-3", EdgeRelation.CHALLENGES))
    store.insert_edge(_edge("edge-4", "unit-3", "unit-4", EdgeRelation.REFINES))
    store.insert_edge(_edge("edge-5", "unit-2", "unit-4", EdgeRelation.CHALLENGES))

    result = store.edge_relation_distribution()

    assert result == {
        EdgeRelation.BUILDS_ON: 2,
        EdgeRelation.CHALLENGES: 2,
        EdgeRelation.REFINES: 1,
    }


def test_edge_relation_distribution_sorted_by_count_descending(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha"))
    store.insert_unit(_unit("unit-2", "Beta"))
    store.insert_unit(_unit("unit-3", "Gamma"))

    # Create edges with different frequencies
    store.insert_edge(_edge("edge-1", "unit-1", "unit-2", EdgeRelation.RELATES_TO))
    store.insert_edge(_edge("edge-2", "unit-2", "unit-3", EdgeRelation.RELATES_TO))
    store.insert_edge(_edge("edge-3", "unit-1", "unit-3", EdgeRelation.RELATES_TO))
    store.insert_edge(_edge("edge-4", "unit-2", "unit-1", EdgeRelation.BUILDS_ON))
    store.insert_edge(_edge("edge-5", "unit-3", "unit-1", EdgeRelation.BUILDS_ON))
    store.insert_edge(_edge("edge-6", "unit-1", "unit-2", EdgeRelation.REFINES))

    result = store.edge_relation_distribution()

    # Verify ordering: RELATES_TO (3), BUILDS_ON (2), REFINES (1)
    assert list(result.keys()) == [
        EdgeRelation.RELATES_TO,
        EdgeRelation.BUILDS_ON,
        EdgeRelation.REFINES,
    ]
    assert result == {
        EdgeRelation.RELATES_TO: 3,
        EdgeRelation.BUILDS_ON: 2,
        EdgeRelation.REFINES: 1,
    }


def test_edge_relation_distribution_alphabetical_order_for_ties(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha"))
    store.insert_unit(_unit("unit-2", "Beta"))
    store.insert_unit(_unit("unit-3", "Gamma"))

    # Create edges with same frequency
    store.insert_edge(_edge("edge-1", "unit-1", "unit-2", EdgeRelation.REFINES))
    store.insert_edge(_edge("edge-2", "unit-2", "unit-3", EdgeRelation.BUILDS_ON))
    store.insert_edge(_edge("edge-3", "unit-1", "unit-3", EdgeRelation.CHALLENGES))

    result = store.edge_relation_distribution()

    # All have frequency 1, should be alphabetically sorted by enum value
    assert list(result.keys()) == [
        EdgeRelation.BUILDS_ON,
        EdgeRelation.CHALLENGES,
        EdgeRelation.REFINES,
    ]
    assert result == {
        EdgeRelation.BUILDS_ON: 1,
        EdgeRelation.CHALLENGES: 1,
        EdgeRelation.REFINES: 1,
    }


def test_edge_relation_distribution_comprehensive_relation_types(store: Store):
    store.insert_unit(_unit("unit-1", "Alpha"))
    store.insert_unit(_unit("unit-2", "Beta"))

    # Test various relation types
    store.insert_edge(_edge("edge-1", "unit-1", "unit-2", EdgeRelation.BUILDS_ON))
    store.insert_edge(_edge("edge-2", "unit-1", "unit-2", EdgeRelation.CHALLENGES))
    store.insert_edge(_edge("edge-3", "unit-1", "unit-2", EdgeRelation.REFINES))
    store.insert_edge(_edge("edge-4", "unit-1", "unit-2", EdgeRelation.DISCOVERS))
    store.insert_edge(_edge("edge-5", "unit-1", "unit-2", EdgeRelation.REPLICATES))
    store.insert_edge(_edge("edge-6", "unit-1", "unit-2", EdgeRelation.INSPIRES))
    store.insert_edge(_edge("edge-7", "unit-1", "unit-2", EdgeRelation.DERIVES_FROM))
    store.insert_edge(_edge("edge-8", "unit-1", "unit-2", EdgeRelation.RELATES_TO))
    store.insert_edge(_edge("edge-9", "unit-1", "unit-2", EdgeRelation.CONTAINS))
    store.insert_edge(_edge("edge-10", "unit-1", "unit-2", EdgeRelation.REFERENCES))

    result = store.edge_relation_distribution()

    assert result == {
        EdgeRelation.BUILDS_ON: 1,
        EdgeRelation.CHALLENGES: 1,
        EdgeRelation.CONTAINS: 1,
        EdgeRelation.DERIVES_FROM: 1,
        EdgeRelation.DISCOVERS: 1,
        EdgeRelation.INSPIRES: 1,
        EdgeRelation.REFERENCES: 1,
        EdgeRelation.REFINES: 1,
        EdgeRelation.RELATES_TO: 1,
        EdgeRelation.REPLICATES: 1,
    }
    assert len(result) == 10
