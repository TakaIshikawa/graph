from __future__ import annotations

import pytest

from graph.rag import build_citation_trails
from graph.rag.citation_trails import build_citation_trails as imported_builder
from graph.types.enums import ContentType, EdgeRelation, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit


def _unit(unit_id: str, title: str, *, metadata: dict | None = None) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=unit_id,
        source_entity_type="note",
        title=title,
        content=f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
    )


def _edge(from_id: str, to_id: str, relation: EdgeRelation = EdgeRelation.REFERENCES) -> KnowledgeEdge:
    return KnowledgeEdge(from_unit_id=from_id, to_unit_id=to_id, relation=relation)


def test_build_citation_trails_from_units_and_edges_with_limits():
    results = [
        _unit("root", "Root", metadata={"doi": "10.1/example"}),
        _unit("support", "Support", metadata={"url": "https://example.test/support"}),
        _unit("related", "Related"),
    ]
    edges = [
        _edge("root", "support", EdgeRelation.REFERENCES),
        _edge("support", "related", EdgeRelation.RELATES_TO),
    ]

    trails = build_citation_trails(results, edges, max_depth=2, max_trails=2)

    assert trails == [
        {
            "root": {
                "id": "related",
                "title": "Related",
                "source_project": "max",
                "citation": {"source_id": "related", "source_entity_type": "note"},
            },
            "depth": 1,
            "path": [
                {
                    "from": {
                        "id": "support",
                        "title": "Support",
                        "source_project": "max",
                        "citation": {
                            "url": "https://example.test/support",
                            "source_id": "support",
                            "source_entity_type": "note",
                        },
                    },
                    "relation": "relates_to",
                    "to": {
                        "id": "related",
                        "title": "Related",
                        "source_project": "max",
                        "citation": {
                            "source_id": "related",
                            "source_entity_type": "note",
                        },
                    },
                }
            ],
        },
        {
            "root": {
                "id": "root",
                "title": "Root",
                "source_project": "max",
                "citation": {
                    "doi": "10.1/example",
                    "source_id": "root",
                    "source_entity_type": "note",
                },
            },
            "depth": 1,
            "path": [
                {
                    "from": {
                        "id": "root",
                        "title": "Root",
                        "source_project": "max",
                        "citation": {
                            "doi": "10.1/example",
                            "source_id": "root",
                            "source_entity_type": "note",
                        },
                    },
                    "relation": "references",
                    "to": {
                        "id": "support",
                        "title": "Support",
                        "source_project": "max",
                        "citation": {
                            "url": "https://example.test/support",
                            "source_id": "support",
                            "source_entity_type": "note",
                        },
                    },
                }
            ],
        },
    ]


def test_build_citation_trails_accepts_dict_results_and_endpoint_units():
    results = [{"unit": {"id": "a", "title": "A", "source_project": "max", "metadata": {"citation_key": "A2026"}}}]
    edges = [
        {
            "from_unit": {"id": "a", "title": "A", "source_project": "max"},
            "to_unit": {"id": "b", "title": "B", "source_project": "presence"},
            "relation": "supports",
        }
    ]

    trails = build_citation_trails(results, edges)

    assert trails[0]["root"]["citation"] == {"citation_key": "A2026"}
    assert trails[0]["path"][0]["to"]["title"] == "B"
    assert trails[0]["path"][0]["relation"] == "supports"


def test_build_citation_trails_validates_limits():
    with pytest.raises(ValueError, match="max_depth must be a positive integer"):
        build_citation_trails([], [], max_depth=0)
    with pytest.raises(ValueError, match="max_trails must be a non-negative integer"):
        build_citation_trails([], [], max_trails=-1)
    assert build_citation_trails([], [], max_trails=0) == []
    assert imported_builder is build_citation_trails
