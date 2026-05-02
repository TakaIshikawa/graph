from __future__ import annotations

import json
from datetime import date, datetime, timezone

from graph.export import export_units_to_jsonl
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(unit_id: str = "unit-a") -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="insight",
        title="Alpha",
        content="Alpha content",
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
        embedding=[0.1, 0.2, 0.3],
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
    )


def records(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines()]


def test_export_units_to_jsonl_returns_one_json_object_per_unit_line():
    text = export_units_to_jsonl([unit("unit-b"), unit("unit-a")])

    lines = text.splitlines()
    parsed = records(text)

    assert len(lines) == 2
    assert [item["id"] for item in parsed] == ["unit-b", "unit-a"]
    assert text.endswith("\n")


def test_export_units_to_jsonl_serializes_json_compatible_values_without_embedding():
    text = export_units_to_jsonl([unit()])
    parsed = records(text)[0]

    assert parsed == {
        "id": "unit-a",
        "source_project": "max",
        "source_id": "source-unit-a",
        "source_entity_type": "insight",
        "title": "Alpha",
        "content": "Alpha content",
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
    assert "embedding" not in parsed


def test_export_units_to_jsonl_includes_embedding_only_when_requested():
    without_embedding = records(export_units_to_jsonl([unit()]))[0]
    with_embedding = records(export_units_to_jsonl([unit()], include_embedding=True))[0]

    assert "embedding" not in without_embedding
    assert with_embedding["embedding"] == [0.1, 0.2, 0.3]


def test_export_units_to_jsonl_writes_same_content_returned(tmp_path):
    path = tmp_path / "nested" / "units.jsonl"

    text = export_units_to_jsonl([unit()], path)

    assert path.read_text(encoding="utf-8") == text
