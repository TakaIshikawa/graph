from __future__ import annotations

import json
from datetime import date, datetime, timezone

from graph.export.json import export_units_to_json
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit


UNIT_TIME = datetime(2026, 5, 1, 10, 15, tzinfo=timezone.utc)


def unit(
    unit_id: str,
    *,
    source_project: SourceProject | str = SourceProject.MAX,
    source_id: str | None = None,
    title: str | None = None,
    content_type: ContentType = ContentType.FINDING,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=source_id or f"source-{unit_id}",
        source_entity_type="insight",
        title=title or f"Title {unit_id}",
        content=f"Content {unit_id}",
        content_type=content_type,
        tags=tags if tags is not None else ["storage", "solar"],
        confidence=0.8,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UNIT_TIME,
        metadata=metadata if metadata is not None else {},
        embedding=[0.1, 0.2],
    )


def test_export_units_to_json_handles_empty_collection():
    text = export_units_to_json([])
    parsed = json.loads(text)

    assert parsed == []


def test_export_units_to_json_exports_single_unit_as_object():
    text = export_units_to_json(unit("unit-a"))
    parsed = json.loads(text)

    assert isinstance(parsed, dict)
    assert parsed["id"] == "unit-a"
    assert parsed["title"] == "Title unit-a"


def test_export_units_to_json_exports_multiple_units_as_array():
    text = export_units_to_json([unit("unit-a"), unit("unit-b")])
    parsed = json.loads(text)

    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert [item["id"] for item in parsed] == ["unit-a", "unit-b"]


def test_export_units_to_json_includes_all_unit_fields_without_embedding():
    text = export_units_to_json(unit("unit-a"))
    parsed = json.loads(text)

    assert list(parsed.keys()) == [
        "confidence",
        "content",
        "content_type",
        "created_at",
        "id",
        "ingested_at",
        "metadata",
        "source_entity_type",
        "source_id",
        "source_project",
        "tags",
        "title",
        "updated_at",
        "utility_score",
    ]
    assert "embedding" not in parsed


def test_export_units_to_json_serializes_all_field_types():
    text = export_units_to_json(
        unit(
            "unit-a",
            source_project=SourceProject.PINBOARD,
            content_type=ContentType.IDEA,
            metadata={
                "reviewed_at": UNIT_TIME,
                "reviewed_on": date(2026, 5, 1),
                "project": SourceProject.MAX,
                "nested": {"type": ContentType.ARTIFACT},
            },
        )
    )
    parsed = json.loads(text)

    assert parsed == {
        "id": "unit-a",
        "source_project": "pinboard",
        "source_id": "source-unit-a",
        "source_entity_type": "insight",
        "title": "Title unit-a",
        "content": "Content unit-a",
        "content_type": "idea",
        "metadata": {
            "nested": {"type": "artifact"},
            "project": "max",
            "reviewed_at": "2026-05-01T10:15:00+00:00",
            "reviewed_on": "2026-05-01",
        },
        "tags": ["storage", "solar"],
        "confidence": 0.8,
        "utility_score": 0.6,
        "created_at": "2026-05-01T10:15:00+00:00",
        "ingested_at": "2026-05-01T10:15:00+00:00",
        "updated_at": "2026-05-01T10:15:00+00:00",
    }


def test_export_units_to_json_filters_metadata_by_keys():
    text = export_units_to_json(
        unit(
            "unit-a",
            metadata={
                "alpha": "value-alpha",
                "beta": "value-beta",
                "gamma": "value-gamma",
            },
        ),
        metadata_keys=["alpha", "gamma"],
    )
    parsed = json.loads(text)

    assert parsed["metadata"] == {
        "alpha": "value-alpha",
        "gamma": "value-gamma",
    }


def test_export_units_to_json_filters_metadata_skips_missing_keys():
    text = export_units_to_json(
        unit("unit-a", metadata={"alpha": "value-alpha"}),
        metadata_keys=["alpha", "missing"],
    )
    parsed = json.loads(text)

    assert parsed["metadata"] == {"alpha": "value-alpha"}


def test_export_units_to_json_filters_metadata_empty_when_no_keys_match():
    text = export_units_to_json(
        unit("unit-a", metadata={"alpha": "value-alpha"}),
        metadata_keys=["missing"],
    )
    parsed = json.loads(text)

    assert parsed["metadata"] == {}


def test_export_units_to_json_filters_metadata_for_multiple_units():
    text = export_units_to_json(
        [
            unit("unit-a", metadata={"alpha": "a1", "beta": "b1"}),
            unit("unit-b", metadata={"alpha": "a2", "beta": "b2", "gamma": "g2"}),
        ],
        metadata_keys=["alpha"],
    )
    parsed = json.loads(text)

    assert parsed[0]["metadata"] == {"alpha": "a1"}
    assert parsed[1]["metadata"] == {"alpha": "a2"}


def test_export_units_to_json_writes_same_content_to_file(tmp_path):
    path = tmp_path / "nested" / "units.json"

    text = export_units_to_json([unit("unit-a")], path)

    assert path.read_text(encoding="utf-8") == text


def test_export_units_to_json_produces_valid_json():
    text = export_units_to_json([unit("unit-a"), unit("unit-b")])

    # Should not raise
    json.loads(text)


def test_export_units_to_json_sorts_keys_for_deterministic_output():
    text = export_units_to_json(unit("unit-a"))

    # Keys should be alphabetically sorted in JSON output
    assert '"confidence":' in text
    assert text.index('"confidence":') < text.index('"content":')
    assert text.index('"content":') < text.index('"content_type":')


def test_export_units_to_json_handles_none_values():
    test_unit = unit("unit-a")
    test_unit.confidence = None
    test_unit.utility_score = None

    text = export_units_to_json(test_unit)
    parsed = json.loads(text)

    assert parsed["confidence"] is None
    assert parsed["utility_score"] is None


def test_export_units_to_json_handles_empty_metadata_and_tags():
    text = export_units_to_json(unit("unit-a", metadata={}, tags=[]))
    parsed = json.loads(text)

    assert parsed["metadata"] == {}
    assert parsed["tags"] == []
