from __future__ import annotations

import csv
from datetime import datetime, timezone

from graph.export import export_graph_edges_csv
from graph.export.csv_edges import FIELDNAMES
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def edge(
    edge_id: str,
    source_id: str,
    target_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    *,
    source: EdgeSource = EdgeSource.INFERRED,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=source_id,
        to_unit_id=target_id,
        relation=relation,
        source=source,
        metadata=metadata or {},
        created_at=EDGE_TIME,
    )


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_export_graph_edges_csv_writes_stable_edge_list(tmp_path):
    path = tmp_path / "edges.csv"

    stats = export_graph_edges_csv(
        [
            edge(
                "edge-b",
                "unit-b",
                "unit-c",
                EdgeRelation.REFERENCES,
                source=EdgeSource.SOURCE,
                metadata={"label": "Cites"},
            ),
            edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        ],
        path,
    )

    fieldnames, exported_rows = rows(path)

    assert fieldnames == FIELDNAMES
    assert [row["source_id"] for row in exported_rows] == ["unit-a", "unit-b"]
    assert exported_rows[0]["target_id"] == "unit-b"
    assert exported_rows[0]["relationship"] == "builds_on"
    assert exported_rows[0]["type"] == "builds_on"
    assert exported_rows[0]["label"] == "builds_on"
    assert exported_rows[1]["label"] == "Cites"
    assert exported_rows[1]["source_adapter"] == "source"
    assert exported_rows[1]["created_at"] == EDGE_TIME.isoformat()
    assert stats == {
        "path": str(path),
        "edges_exported": 2,
        "bytes_written": path.stat().st_size,
        "fieldnames": FIELDNAMES,
    }


def test_export_graph_edges_csv_returns_csv_text():
    text = export_graph_edges_csv([edge("edge-a", "unit-a", "unit-b")])

    assert isinstance(text, str)
    assert text.splitlines()[0] == ",".join(FIELDNAMES)
    assert "unit-a,unit-b,relates_to,relates_to,relates_to,inferred" in text


def test_export_graph_edges_csv_handles_empty_edges(tmp_path):
    path = tmp_path / "empty.csv"

    export_graph_edges_csv([], path)

    assert path.read_text(encoding="utf-8") == ",".join(FIELDNAMES) + "\n"
