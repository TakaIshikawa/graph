from __future__ import annotations

import csv
from io import StringIO

from graph.export import export_edge_metadata_key_coverage_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(edge_id: str, *, relation=EdgeRelation.RELATES_TO, source=EdgeSource.INFERRED, metadata: dict | None = None):
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=source,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_metadata_key_coverage_csv_empty_input_has_header_only():
    assert export_edge_metadata_key_coverage_csv([]) == (
        "relation,source,metadata_key,edge_count,edges_with_key,coverage_percent\n"
    )


def test_edge_metadata_key_coverage_csv_counts_presence_by_relation_and_source():
    text = export_edge_metadata_key_coverage_csv(
        [
            edge("a", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE, metadata={"kind": "citation"}),
            edge("b", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE, metadata={"kind": "", "page": 2}),
            edge("c", relation=EdgeRelation.REFERENCES, source=EdgeSource.SOURCE, metadata={}),
            edge("d", relation="custom", source="external", metadata={"kind": "imported"}),
        ]
    )

    assert rows(text) == [
        {"relation": "custom", "source": "external", "metadata_key": "kind", "edge_count": "1", "edges_with_key": "1", "coverage_percent": "100.00"},
        {"relation": "references", "source": "source", "metadata_key": "kind", "edge_count": "3", "edges_with_key": "2", "coverage_percent": "66.67"},
        {"relation": "references", "source": "source", "metadata_key": "page", "edge_count": "3", "edges_with_key": "1", "coverage_percent": "33.33"},
    ]


def test_edge_metadata_key_coverage_csv_accepts_mappings_and_unknown_fallbacks(tmp_path):
    path = tmp_path / "edge-coverage.csv"
    edges = [{"relation": "", "source": None, "metadata": {"note": None}}, {"metadata": {}}]

    expected = export_edge_metadata_key_coverage_csv(edges)
    stats = export_edge_metadata_key_coverage_csv(edges, path)

    assert rows(expected) == [
        {"relation": "Unknown", "source": "Unknown", "metadata_key": "note", "edge_count": "2", "edges_with_key": "1", "coverage_percent": "50.00"}
    ]
    assert path.read_text(encoding="utf-8") == expected
    assert stats["edge_count"] == 2
    assert stats["rows_exported"] == 1
