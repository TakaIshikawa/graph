from __future__ import annotations

import csv
from io import StringIO

import pytest

from graph.export.edge_metadata_density_csv import export_edge_metadata_density_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    relation: EdgeRelation | str | None = EdgeRelation.REFERENCES,
    *,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=EdgeSource.SOURCE,
        weight=1.0,
        metadata=metadata or {},
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_metadata_density_groups_by_relation():
    text = export_edge_metadata_density_csv(
        [
            edge("b", EdgeRelation.REFERENCES, metadata={"source_id": "doc-1", "page": 5}),
            edge("a", EdgeRelation.RELATES_TO, metadata={"source_id": "doc-2"}),
            edge("c", EdgeRelation.REFERENCES, metadata={}),
            edge("d", EdgeRelation.REFERENCES, metadata={"source_id": "doc-3"}),
        ]
    )

    assert text.splitlines()[0] == (
        "relation,edge_count,edges_with_metadata,metadata_coverage_percent,"
        "distinct_metadata_keys,average_keys_per_edge,top_metadata_keys"
    )
    assert rows(text) == [
        {
            "relation": "references",
            "edge_count": "3",
            "edges_with_metadata": "2",
            "metadata_coverage_percent": "66.67",
            "distinct_metadata_keys": "2",
            "average_keys_per_edge": "1.00",
            "top_metadata_keys": "source_id (2); page (1)",
        },
        {
            "relation": "relates_to",
            "edge_count": "1",
            "edges_with_metadata": "1",
            "metadata_coverage_percent": "100.00",
            "distinct_metadata_keys": "1",
            "average_keys_per_edge": "1.00",
            "top_metadata_keys": "source_id (1)",
        },
    ]


def test_edge_metadata_density_normalizes_blank_relation_and_keys():
    text = export_edge_metadata_density_csv(
        [
            edge("a", None, metadata={"": 1, " source\nurl ": "https://example.test"}),
            edge("b", " \n\t ", metadata={}),
            edge("c", "  Cites\nStrongly  ", metadata={"source url": "https://example.test/2"}),
        ]
    )

    assert rows(text) == [
        {
            "relation": "Cites Strongly",
            "edge_count": "1",
            "edges_with_metadata": "1",
            "metadata_coverage_percent": "100.00",
            "distinct_metadata_keys": "1",
            "average_keys_per_edge": "1.00",
            "top_metadata_keys": "source url (1)",
        },
        {
            "relation": "Unknown",
            "edge_count": "2",
            "edges_with_metadata": "1",
            "metadata_coverage_percent": "50.00",
            "distinct_metadata_keys": "1",
            "average_keys_per_edge": "0.50",
            "top_metadata_keys": "source url (1)",
        },
    ]


def test_edge_metadata_density_min_edges_filters_and_validates():
    text = export_edge_metadata_density_csv(
        [
            edge("a", "Source A", metadata={"a": 1}),
            edge("b", "Source B", metadata={"b": 1}),
            edge("c", "Source B", metadata={}),
        ],
        min_edges=2,
    )

    assert [row["relation"] for row in rows(text)] == ["Source B"]

    with pytest.raises(ValueError, match="min_edges"):
        export_edge_metadata_density_csv([], min_edges=0)
    with pytest.raises(ValueError, match="min_edges"):
        export_edge_metadata_density_csv([], min_edges=True)


def test_edge_metadata_density_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "reports" / "edge-density.csv"
    edges = [edge("a", "Source A", metadata={"a": 1}), edge("b", "Source A", metadata={})]

    expected = export_edge_metadata_density_csv(edges)
    stats = export_edge_metadata_density_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 2,
        "relation_count": 1,
        "rows_exported": 1,
        "min_edges": 1,
        "bytes_written": path.stat().st_size,
    }


def test_edge_metadata_density_is_deterministic_for_reversed_input():
    edges = [
        edge("a", "Source B", metadata={"z": 1}),
        edge("b", "Source A", metadata={"a": 1, "z": 2}),
        edge("c", "Source A", metadata={"a": 3}),
    ]

    assert export_edge_metadata_density_csv(edges) == export_edge_metadata_density_csv(reversed(edges))
