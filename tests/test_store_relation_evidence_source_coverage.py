from __future__ import annotations

from graph.store.relation_evidence_source_coverage import relation_evidence_source_coverage
from graph.types.enums import EdgeRelation
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _edge(edge_id: str, metadata: dict):
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id="a",
        to_unit_id="b",
        relation=EdgeRelation.RELATES_TO,
        metadata=metadata,
    )


def test_relation_evidence_source_coverage_classifies_none_single_and_multi_source():
    units = [
        KnowledgeUnit(
            id="unit-a",
            source_project="notes",
            source_id="source-a",
            source_entity_type="note",
            title="A",
            content="",
        )
    ]
    rows = relation_evidence_source_coverage(
        [
            _edge("none", {}),
            _edge("single", {"evidence": [{"source_id": "s1"}, {"source_id": "s1"}]}),
            _edge("multi", {"evidence": [{"source": {"id": "s1"}}, {"unit_id": "unit-a"}]}),
        ],
        units=units,
    )

    assert rows == [
        {
            "relation_id": "multi",
            "relation": "relates_to",
            "evidence_count": 2,
            "distinct_source_count": 2,
            "sources": ["s1", "source-a"],
            "coverage_status": "multi_source",
        },
        {
            "relation_id": "none",
            "relation": "relates_to",
            "evidence_count": 0,
            "distinct_source_count": 0,
            "sources": [],
            "coverage_status": "none",
        },
        {
            "relation_id": "single",
            "relation": "relates_to",
            "evidence_count": 2,
            "distinct_source_count": 1,
            "sources": ["s1"],
            "coverage_status": "single_source",
        },
    ]
