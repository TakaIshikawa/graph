"""Tests for JSON export adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from graph.exports import export_units_to_json, get_exporter, list_exporters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, 30, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 2, 8, 30, 45, tzinfo=timezone.utc)


def _unit(
    unit_id: str,
    title: str,
    content: str = "",
    *,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    confidence: float | None = 0.8,
    utility_score: float | None = 0.6,
) -> KnowledgeUnit:
    """Create a test unit with sensible defaults."""
    return KnowledgeUnit(
        id=unit_id,
        source_project=SourceProject.MAX,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content or f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
        confidence=confidence,
        utility_score=utility_score,
        created_at=created_at or UNIT_TIME,
        ingested_at=created_at or UNIT_TIME,
        updated_at=updated_at or UPDATED_TIME,
    )


def test_export_units_to_json_full_mode_generates_valid_json():
    """Test that full mode generates valid JSON."""
    units = [_unit("unit-1", "Test Unit")]
    json_str = export_units_to_json(units, mode="full")

    # Should be valid JSON
    data = json.loads(json_str)
    assert isinstance(data, list)
    assert len(data) == 1


def test_export_units_to_json_compact_mode_generates_valid_json():
    """Test that compact mode generates valid JSON."""
    units = [_unit("unit-1", "Test Unit")]
    json_str = export_units_to_json(units, mode="compact")

    # Should be valid JSON
    data = json.loads(json_str)
    assert isinstance(data, list)
    assert len(data) == 1


def test_export_units_to_json_full_mode_includes_all_fields():
    """Test that full mode includes all unit fields."""
    unit = _unit(
        "unit-alpha",
        "Alpha Unit",
        "Alpha content",
        tags=["ai", "research"],
        metadata={"author": "Test"},
        confidence=0.9,
        utility_score=0.7,
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    assert len(data) == 1
    unit_dict = data[0]

    assert unit_dict["id"] == "unit-alpha"
    assert unit_dict["source_project"] == "max"
    assert unit_dict["source_id"] == "source-unit-alpha"
    assert unit_dict["source_entity_type"] == "note"
    assert unit_dict["title"] == "Alpha Unit"
    assert unit_dict["content"] == "Alpha content"
    assert unit_dict["content_type"] == "insight"
    assert unit_dict["metadata"] == {"author": "Test"}
    assert unit_dict["tags"] == ["ai", "research"]
    assert unit_dict["confidence"] == 0.9
    assert unit_dict["utility_score"] == 0.7
    assert "created_at" in unit_dict
    assert "ingested_at" in unit_dict
    assert "updated_at" in unit_dict


def test_export_units_to_json_compact_mode_only_includes_id_title_content():
    """Test that compact mode only includes id, title, and content."""
    unit = _unit(
        "unit-beta",
        "Beta Unit",
        "Beta content",
        tags=["ai"],
        metadata={"author": "Test"},
        confidence=0.9,
    )
    json_str = export_units_to_json([unit], mode="compact")
    data = json.loads(json_str)

    assert len(data) == 1
    unit_dict = data[0]

    # Should only have these three fields
    assert set(unit_dict.keys()) == {"id", "title", "content"}
    assert unit_dict["id"] == "unit-beta"
    assert unit_dict["title"] == "Beta Unit"
    assert unit_dict["content"] == "Beta content"


def test_export_units_to_json_multiple_units():
    """Test exporting multiple units."""
    units = [
        _unit("unit-1", "First", "First content"),
        _unit("unit-2", "Second", "Second content"),
        _unit("unit-3", "Third", "Third content"),
    ]
    json_str = export_units_to_json(units, mode="compact")
    data = json.loads(json_str)

    assert len(data) == 3
    assert data[0]["id"] == "unit-1"
    assert data[1]["id"] == "unit-2"
    assert data[2]["id"] == "unit-3"


def test_export_units_to_json_empty_list():
    """Test exporting empty unit list."""
    json_str = export_units_to_json([], mode="full")
    data = json.loads(json_str)

    assert data == []


def test_export_units_to_json_pretty_print_formats_with_indentation():
    """Test that pretty=True adds indentation."""
    units = [_unit("unit-1", "Test")]
    json_str = export_units_to_json(units, mode="compact", pretty=True)

    # Pretty-printed JSON should have newlines and indentation
    assert "\n" in json_str
    assert "  " in json_str  # Should have indentation
    # Should still be valid JSON
    data = json.loads(json_str)
    assert len(data) == 1


def test_export_units_to_json_no_pretty_print_is_compact():
    """Test that pretty=False produces compact JSON."""
    units = [_unit("unit-1", "Test")]
    json_str = export_units_to_json(units, mode="compact", pretty=False)

    # Compact JSON should be a single line (or minimal formatting)
    lines = [line for line in json_str.split("\n") if line.strip()]
    # Compact JSON is typically one line
    assert len(lines) <= 1


def test_export_units_to_json_serializes_timestamps_as_iso():
    """Test that timestamps are serialized as ISO format strings."""
    unit = _unit(
        "unit-1",
        "Time Test",
        created_at=UNIT_TIME,
        updated_at=UPDATED_TIME,
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    assert data[0]["created_at"] == "2026-05-01T10:15:30+00:00"
    assert data[0]["updated_at"] == "2026-05-02T08:30:45+00:00"
    assert data[0]["ingested_at"] == "2026-05-01T10:15:30+00:00"


def test_export_units_to_json_serializes_enums():
    """Test that enum values are properly serialized."""
    unit = _unit("unit-1", "Enum Test")
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    # Enums should be serialized to their string values
    assert data[0]["source_project"] == "max"
    assert data[0]["content_type"] == "insight"


def test_export_units_to_json_sorts_tags_alphabetically():
    """Test that tags are sorted in the output."""
    unit = _unit("unit-1", "Tag Test", tags=["zeta", "alpha", "beta"])
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    assert data[0]["tags"] == ["alpha", "beta", "zeta"]


def test_export_units_to_json_handles_none_values():
    """Test handling of None values in fields."""
    unit = _unit("unit-1", "None Test", confidence=None, utility_score=None)
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    assert data[0]["confidence"] is None
    assert data[0]["utility_score"] is None


def test_export_units_to_json_handles_empty_metadata():
    """Test handling of empty metadata."""
    unit = _unit("unit-1", "Empty Meta", metadata={})
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    assert data[0]["metadata"] == {}


def test_export_units_to_json_handles_empty_tags():
    """Test handling of empty tags."""
    unit = _unit("unit-1", "No Tags", tags=[])
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    assert data[0]["tags"] == []


def test_export_units_to_json_handles_complex_metadata():
    """Test handling of complex metadata types."""
    unit = _unit(
        "unit-1",
        "Complex Meta",
        metadata={
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert metadata["string"] == "value"
    assert metadata["number"] == 42
    assert metadata["float"] == 3.14
    assert metadata["bool"] is True
    assert metadata["list"] == [1, 2, 3]
    assert metadata["dict"] == {"nested": "value"}


def test_export_units_to_json_round_trip_compatibility():
    """Test that exported JSON can be used to reconstruct units."""
    original_unit = _unit(
        "unit-round-trip",
        "Round Trip Test",
        "Test content",
        tags=["tag1", "tag2"],
        metadata={"key": "value"},
    )
    json_str = export_units_to_json([original_unit], mode="full")
    data = json.loads(json_str)

    # Should be able to reconstruct a KnowledgeUnit from the data
    reconstructed = KnowledgeUnit(**data[0])
    assert reconstructed.id == original_unit.id
    assert reconstructed.title == original_unit.title
    assert reconstructed.content == original_unit.content
    assert reconstructed.tags == sorted(original_unit.tags)
    assert reconstructed.metadata == original_unit.metadata


def test_export_units_to_json_preserves_unicode():
    """Test that Unicode characters are preserved."""
    unit = _unit("unit-1", "Unicode: 你好世界 🌍", "Content: émojis 🎉")
    json_str = export_units_to_json([unit], mode="compact")
    data = json.loads(json_str)

    assert "你好世界" in data[0]["title"]
    assert "🌍" in data[0]["title"]
    assert "émojis" in data[0]["content"]
    assert "🎉" in data[0]["content"]


def test_export_units_to_json_invalid_mode_raises_error():
    """Test that invalid mode raises ValueError."""
    units = [_unit("unit-1", "Test")]

    with pytest.raises(ValueError, match="mode must be 'full' or 'compact'"):
        export_units_to_json(units, mode="invalid")


def test_export_units_to_json_default_mode_is_full():
    """Test that default mode is full."""
    unit = _unit("unit-1", "Default Mode", tags=["test"])
    json_str = export_units_to_json([unit])
    data = json.loads(json_str)

    # Full mode should include all fields
    assert "metadata" in data[0]
    assert "tags" in data[0]
    assert "confidence" in data[0]


def test_export_units_to_json_default_pretty_is_false():
    """Test that default pretty is False (compact)."""
    units = [_unit("unit-1", "Test")]
    json_str = export_units_to_json(units)

    # Should be compact by default (single line or minimal formatting)
    lines = [line for line in json_str.split("\n") if line.strip()]
    assert len(lines) <= 1


def test_json_exporter_registered_in_registry():
    """Test that JSON exporter is registered in the export registry."""
    exporters = list_exporters()
    assert "json" in exporters


def test_get_json_exporter_from_registry():
    """Test retrieving JSON exporter from registry."""
    exporter = get_exporter("json")
    assert exporter is export_units_to_json


def test_json_exporter_works_through_registry():
    """Test that JSON export works when called through registry."""
    from graph.exports import export_units

    units = [_unit("unit-1", "Registry Test")]
    json_str = export_units("json", units, mode="compact")
    data = json.loads(json_str)

    assert len(data) == 1
    assert data[0]["id"] == "unit-1"
    assert data[0]["title"] == "Registry Test"
