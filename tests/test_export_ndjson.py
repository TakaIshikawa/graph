from __future__ import annotations

import json
from datetime import date, datetime, timezone

import pytest

from graph.export import export_graph_ndjson, export_units_to_ndjson
from graph.types.enums import ContentType, EdgeRelation, EdgeSource, SourceProject
from graph.types.models import KnowledgeEdge, KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)
EDGE_TIME = datetime(2026, 5, 1, 12, 30, tzinfo=timezone.utc)


def unit(unit_id: str = "unit-a", *, created_at: datetime = UNIT_TIME) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title=f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=ContentType.FINDING,
        metadata={
            "reviewed_at": UNIT_TIME,
            "reviewed_on": date(2026, 5, 1),
            "project": SourceProject.MAX,
            "nested": {"type": ContentType.IDEA},
        },
        tags=["solar", "storage"],
        confidence=0.8,
        utility_score=0.6,
        created_at=created_at,
        ingested_at=created_at,
        updated_at=created_at,
    )


def edge(edge_id: str = "edge-a", *, created_at: datetime = EDGE_TIME) -> KnowledgeEdge:
    return KnowledgeEdge(
        id=edge_id,
        from_unit_id="unit-a",
        to_unit_id="unit-b",
        relation=EdgeRelation.BUILDS_ON,
        weight=0.75,
        source=EdgeSource.MANUAL,
        metadata={"observed_at": EDGE_TIME, "notes": ["curated"]},
        created_at=created_at,
    )


def records(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]


def test_export_graph_ndjson_emits_one_compact_json_object_per_line():
    text = export_graph_ndjson([unit("unit-a"), unit("unit-b")], [edge()])
    lines = text.splitlines()
    parsed = records(text)

    assert len(lines) == 3
    assert all(json.loads(line) for line in lines)
    assert all("\n" not in line and ": " not in line for line in lines)
    assert text.endswith("\n")
    assert [record["record_type"] for record in parsed] == ["unit", "unit", "edge"]
    assert [record["id"] for record in parsed] == ["unit-a", "unit-b", "edge-a"]


def test_export_graph_ndjson_serializes_unit_and_edge_core_fields():
    parsed = records(export_graph_ndjson(unit("unit-a"), [edge()]))

    assert parsed[0] == {
        "record_type": "unit",
        "id": "unit-a",
        "source_project": "max",
        "source_id": "source-unit-a",
        "source_entity_type": "insight",
        "title": "Title unit-a",
        "content": "Content unit-a",
        "content_type": "finding",
        "metadata": {
            "nested": {"type": "idea"},
            "project": "max",
            "reviewed_at": "2026-05-01T10:15:00+00:00",
            "reviewed_on": "2026-05-01",
        },
        "tags": ["solar", "storage"],
        "confidence": 0.8,
        "utility_score": 0.6,
        "created_at": "2026-05-01T10:15:00+00:00",
        "ingested_at": "2026-05-01T10:15:00+00:00",
        "updated_at": "2026-05-01T10:15:00+00:00",
    }
    assert parsed[1] == {
        "record_type": "edge",
        "id": "edge-a",
        "from_unit_id": "unit-a",
        "to_unit_id": "unit-b",
        "relation": "builds_on",
        "weight": 0.75,
        "source": "manual",
        "metadata": {"notes": ["curated"], "observed_at": "2026-05-01T12:30:00+00:00"},
        "created_at": "2026-05-01T12:30:00+00:00",
    }


def test_export_graph_ndjson_sorts_records_deterministically():
    text_a = export_graph_ndjson(
        [
            unit("unit-b", created_at=datetime(2026, 5, 1, 10, 16, tzinfo=timezone.utc)),
            unit("unit-a", created_at=UNIT_TIME),
        ],
        [edge("edge-a", created_at=EDGE_TIME)],
    )
    text_b = export_graph_ndjson(
        [
            unit("unit-a", created_at=UNIT_TIME),
            unit("unit-b", created_at=datetime(2026, 5, 1, 10, 16, tzinfo=timezone.utc)),
        ],
        [edge("edge-a", created_at=EDGE_TIME)],
    )

    assert text_a == text_b
    assert [record["id"] for record in records(text_a)] == ["unit-a", "unit-b", "edge-a"]


def test_export_graph_ndjson_filters_record_type_and_rejects_invalid_type():
    assert [record["record_type"] for record in records(export_graph_ndjson([unit()], [edge()], record_type="units"))] == [
        "unit"
    ]
    assert [record["record_type"] for record in records(export_graph_ndjson([unit()], [edge()], record_type="edges"))] == [
        "edge"
    ]

    with pytest.raises(ValueError, match="record_type"):
        export_graph_ndjson([], [], record_type="invalid")  # type: ignore[arg-type]


def test_export_graph_ndjson_empty_graph_is_empty_string_and_file(tmp_path):
    path = tmp_path / "nested" / "graph.ndjson"

    text = export_graph_ndjson([], [], path)

    assert text == ""
    assert path.read_text(encoding="utf-8") == ""


def test_export_graph_ndjson_writes_same_content_returned(tmp_path):
    path = tmp_path / "nested" / "graph.ndjson"

    text = export_graph_ndjson([unit()], [edge()], path)

    assert path.read_text(encoding="utf-8") == text
    assert records(path.read_text(encoding="utf-8")) == records(text)


def test_export_units_to_ndjson_exports_unit_records_only():
    parsed = records(export_units_to_ndjson([unit("unit-a")]))

    assert [record["record_type"] for record in parsed] == ["unit"]
    assert parsed[0]["id"] == "unit-a"
