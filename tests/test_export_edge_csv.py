from __future__ import annotations

import csv
import json
from datetime import datetime, timezone

from graph.export import export_edges_to_csv
from graph.types.enums import EdgeRelation, EdgeSource
from graph.types.models import KnowledgeEdge


def edge(
    edge_id: str,
    from_unit_id: str,
    to_unit_id: str,
    relation: EdgeRelation = EdgeRelation.RELATES_TO,
    *,
    weight: float = 1.0,
    source: EdgeSource = EdgeSource.INFERRED,
    metadata: dict | None = None,
) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id=from_unit_id,
        to_unit_id=to_unit_id,
        relation=relation,
        weight=weight,
        source=source,
        metadata=metadata or {},
        created_at=datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc),
    )


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_export_edges_csv_writes_stable_headers_and_stats(tmp_path):
    path = tmp_path / "edges.csv"
    stats = export_edges_to_csv(
        [
            edge(
                "edge-1",
                "unit-a",
                "unit-b",
                EdgeRelation.BUILDS_ON,
                weight=0.75,
                source=EdgeSource.MANUAL,
                metadata={"note": "curated"},
            )
        ],
        path,
    )

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "id",
        "from_unit_id",
        "to_unit_id",
        "relation",
        "weight",
        "source",
        "created_at",
        "metadata_json",
    ]
    assert stats == {
        "path": str(path),
        "edges_scanned": 1,
        "edges_exported": 1,
        "bytes_written": path.stat().st_size,
    }
    assert rows[0]["id"] == "edge-1"


def test_export_edges_csv_serializes_enums_and_metadata_json(tmp_path):
    path = tmp_path / "edges.csv"
    export_edges_to_csv(
        [
            edge(
                "edge-1",
                "unit-a",
                "unit-b",
                EdgeRelation.REFERENCES,
                source=EdgeSource.SOURCE,
                metadata={"z": 2, "a": {"nested": True}},
            )
        ],
        path,
    )

    row = read_rows(path)[0]

    assert row["relation"] == "references"
    assert row["source"] == "source"
    assert row["metadata_json"] == '{"a":{"nested":true},"z":2}'
    assert json.loads(row["metadata_json"]) == {"a": {"nested": True}, "z": 2}


def test_export_edges_csv_sorts_rows_deterministically(tmp_path):
    path = tmp_path / "edges.csv"
    export_edges_to_csv(
        [
            edge("edge-c", "unit-b", "unit-a", EdgeRelation.RELATES_TO),
            edge("edge-b", "unit-a", "unit-b", EdgeRelation.REFERENCES),
            edge("edge-a", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
            edge("edge-d", "unit-a", "unit-b", EdgeRelation.BUILDS_ON),
        ],
        path,
    )

    assert [row["id"] for row in read_rows(path)] == [
        "edge-a",
        "edge-d",
        "edge-b",
        "edge-c",
    ]


def test_export_edges_csv_can_omit_metadata(tmp_path):
    path = tmp_path / "edges.csv"
    export_edges_to_csv(
        [edge("edge-1", "unit-a", "unit-b", metadata={"note": "hidden"})],
        path,
        include_metadata=False,
    )

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "id",
        "from_unit_id",
        "to_unit_id",
        "relation",
        "weight",
        "source",
        "created_at",
    ]
    assert "metadata_json" not in rows[0]


def test_export_edges_csv_empty_export_writes_header_only(tmp_path):
    path = tmp_path / "empty.csv"
    stats = export_edges_to_csv([], path)

    text = path.read_text(encoding="utf-8")

    assert text == "id,from_unit_id,to_unit_id,relation,weight,source,created_at,metadata_json\n"
    assert stats == {
        "path": str(path),
        "edges_scanned": 0,
        "edges_exported": 0,
        "bytes_written": len(text.encode("utf-8")),
    }
