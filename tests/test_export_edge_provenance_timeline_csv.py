from __future__ import annotations

import csv
from io import StringIO

from graph.export.edge_provenance_timeline_csv import export_edge_provenance_timeline_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    *,
    relation: EdgeRelation | str = EdgeRelation.RELATES_TO,
    source: EdgeSource | str = EdgeSource.INFERRED,
    weight: object = 1.0,
    created_at: object = None,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge.model_construct(
        id=edge_id,
        from_unit_id=f"from-{edge_id}",
        to_unit_id=f"to-{edge_id}",
        relation=relation,
        source=source,
        weight=weight,
        metadata=metadata or {},
        created_at=created_at,
    )


def rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(text)))


def test_edge_provenance_timeline_csv_empty_input_has_header_only():
    assert export_edge_provenance_timeline_csv([]) == (
        "edge_id,from_unit_id,to_unit_id,relation,source,source_project,source_entity_type,"
        "provenance_date,provenance_key,weight,metadata_key_count\n"
    )


def test_edge_provenance_timeline_csv_prefers_metadata_provenance_dates():
    text = export_edge_provenance_timeline_csv(
        [
            edge(
                "a",
                relation=EdgeRelation.REFERENCES,
                source=EdgeSource.SOURCE,
                weight=2.5,
                created_at="2024-01-01",
                metadata={
                    "source_project": "Project A",
                    "source_entity_type": "issue",
                    "observed_date": "2024-01-03T08:00:00Z",
                    "date": "2024-01-04",
                },
            )
        ]
    )

    assert rows(text) == [
        {
            "edge_id": "a",
            "from_unit_id": "from-a",
            "to_unit_id": "to-a",
            "relation": "references",
            "source": "source",
            "source_project": "Project A",
            "source_entity_type": "issue",
            "provenance_date": "2024-01-03",
            "provenance_key": "observed_date",
            "weight": "2.50",
            "metadata_key_count": "4",
        }
    ]


def test_edge_provenance_timeline_csv_falls_back_to_created_at():
    text = export_edge_provenance_timeline_csv(
        [
            edge(
                "fallback",
                relation="custom relation",
                source="custom source",
                created_at="2024-02-10T10:00:00Z",
                metadata={"source_project": "Project B", "observed_at": "not a date"},
            )
        ]
    )

    assert rows(text) == [
        {
            "edge_id": "fallback",
            "from_unit_id": "from-fallback",
            "to_unit_id": "to-fallback",
            "relation": "custom relation",
            "source": "custom source",
            "source_project": "Project B",
            "source_entity_type": "Unknown",
            "provenance_date": "2024-02-10",
            "provenance_key": "created_at",
            "weight": "1.00",
            "metadata_key_count": "2",
        }
    ]


def test_edge_provenance_timeline_csv_sorts_deterministically():
    edges = [
        edge("b", created_at="2024-02-01"),
        edge("a", metadata={"published_at": "2024-01-01"}),
        edge("c", created_at="2024-03-01"),
    ]

    assert export_edge_provenance_timeline_csv(edges) == export_edge_provenance_timeline_csv(reversed(edges))
    assert [row["edge_id"] for row in rows(export_edge_provenance_timeline_csv(edges))] == [
        "a",
        "b",
        "c",
    ]


def test_edge_provenance_timeline_csv_path_mode_writes_same_content_and_stats(tmp_path):
    path = tmp_path / "edge-provenance.csv"
    edges = [edge("a", created_at="2024-01-01")]

    expected = export_edge_provenance_timeline_csv(edges)
    stats = export_edge_provenance_timeline_csv(edges, path)

    assert path.read_text(encoding="utf-8") == expected
    assert stats == {
        "path": str(path),
        "edge_count": 1,
        "rows_exported": 1,
        "bytes_written": path.stat().st_size,
    }
