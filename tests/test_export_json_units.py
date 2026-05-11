"""Tests for JSON export adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel

from graph.exports import export_units_to_json, get_exporter, list_exporters
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 1, 10, 15, 30, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 2, 8, 30, 45, tzinfo=timezone.utc)


class MetadataModel(BaseModel):
    label: str
    count: int


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


# Edge Case Tests


def test_export_units_to_json_empty_collection_full_mode():
    """Test that empty collections produce empty JSON arrays in full mode."""
    json_str = export_units_to_json([], mode="full")
    data = json.loads(json_str)

    assert data == []
    assert isinstance(data, list)


def test_export_units_to_json_empty_collection_compact_mode():
    """Test that empty collections produce empty JSON arrays in compact mode."""
    json_str = export_units_to_json([], mode="compact")
    data = json.loads(json_str)

    assert data == []
    assert isinstance(data, list)


def test_export_units_to_json_metadata_with_none_values():
    """Test that metadata containing None values is properly serialized."""
    unit = _unit(
        "unit-null-meta",
        "Null Metadata Test",
        metadata={
            "key1": "value1",
            "key2": None,
            "key3": "value3",
            "key4": None,
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert metadata["key1"] == "value1"
    assert metadata["key2"] is None
    assert metadata["key3"] == "value3"
    assert metadata["key4"] is None


def test_export_units_to_json_metadata_with_all_none_values():
    """Test metadata dictionary where all values are None."""
    unit = _unit(
        "unit-all-none",
        "All None Metadata",
        metadata={"a": None, "b": None, "c": None},
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert metadata == {"a": None, "b": None, "c": None}


def test_export_units_to_json_deeply_nested_metadata():
    """Test handling of deeply nested metadata structures."""
    unit = _unit(
        "unit-nested",
        "Nested Metadata",
        metadata={
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": {
                                "deep_value": "found it",
                                "deep_number": 42,
                                "deep_list": [1, 2, 3],
                            }
                        }
                    }
                }
            },
            "another_top": {
                "nested_list": [
                    {"item": 1, "data": {"sub": "a"}},
                    {"item": 2, "data": {"sub": "b"}},
                ]
            },
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert metadata["level1"]["level2"]["level3"]["level4"]["level5"]["deep_value"] == "found it"
    assert metadata["level1"]["level2"]["level3"]["level4"]["level5"]["deep_number"] == 42
    assert metadata["level1"]["level2"]["level3"]["level4"]["level5"]["deep_list"] == [1, 2, 3]
    assert metadata["another_top"]["nested_list"][0]["data"]["sub"] == "a"
    assert metadata["another_top"]["nested_list"][1]["data"]["sub"] == "b"


def test_export_units_to_json_full_mode_serializes_nested_non_json_metadata():
    """Test nested non-primitive metadata becomes JSON-safe in full mode."""
    unit = _unit(
        "unit-non-json",
        "Non JSON Metadata",
        metadata={
            "when": UNIT_TIME,
            "kind": ContentType.FINDING,
            "tuple": ("alpha", datetime(2026, 5, 3, 1, 2, tzinfo=timezone.utc)),
            "mapping": {1: {"model": MetadataModel(label="nested", count=2)}},
        },
    )

    data = json.loads(export_units_to_json([unit], mode="full"))

    assert data[0]["metadata"] == {
        "kind": "finding",
        "mapping": {"1": {"model": {"count": 2, "label": "nested"}}},
        "tuple": ["alpha", "2026-05-03T01:02:00+00:00"],
        "when": "2026-05-01T10:15:30+00:00",
    }


def test_export_units_to_json_compact_mode_ignores_non_json_metadata():
    """Test compact mode remains limited to id, title, and content."""
    unit = _unit(
        "unit-compact-non-json",
        "Compact Non JSON",
        metadata={"when": UNIT_TIME, "tuple": ("alpha", "beta")},
    )

    data = json.loads(export_units_to_json([unit], mode="compact"))

    assert data == [
        {
            "id": "unit-compact-non-json",
            "title": "Compact Non JSON",
            "content": "Compact Non JSON content",
        }
    ]


def test_export_units_to_json_unicode_in_all_fields():
    """Test Unicode characters in all text fields."""
    unit = _unit(
        "unit-unicode-全",
        "Title with émojis 🚀 and 中文字符",
        "Content with Arabic: مرحبا, Russian: Привет, Hebrew: שלום, Greek: Γειά σου",
        tags=["tag-日本語", "tag-한글", "tag-🏷️"],
        metadata={
            "field_中文": "值",
            "field_emoji": "🎨🎭🎪",
            "field_mixed": "Mix of ASCII and 日本語 and 🌟",
            "nested": {
                "deeper_unicode": "Ñoño ñandú",
            },
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    # Verify all Unicode is preserved
    assert data[0]["id"] == "unit-unicode-全"
    assert "émojis 🚀" in data[0]["title"]
    assert "中文字符" in data[0]["title"]
    assert "مرحبا" in data[0]["content"]
    assert "Привет" in data[0]["content"]
    assert "שלום" in data[0]["content"]
    assert "Γειά σου" in data[0]["content"]

    # Tags should be sorted, verify they're present
    tags = data[0]["tags"]
    assert "tag-🏷️" in tags
    assert "tag-日本語" in tags
    assert "tag-한글" in tags

    # Metadata Unicode
    metadata = data[0]["metadata"]
    assert metadata["field_中文"] == "值"
    assert metadata["field_emoji"] == "🎨🎭🎪"
    assert "日本語" in metadata["field_mixed"]
    assert metadata["nested"]["deeper_unicode"] == "Ñoño ñandú"


def test_export_units_to_json_very_long_content():
    """Test handling of very long content strings."""
    # Generate a very long content string (>100KB)
    long_content = "A" * 100000 + " " + "B" * 50000
    long_title = "Long Title " + "X" * 10000

    unit = _unit(
        "unit-long",
        long_title,
        long_content,
        metadata={"long_field": "Y" * 50000},
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    # Verify content is fully preserved
    assert data[0]["title"] == long_title
    assert data[0]["content"] == long_content
    assert data[0]["metadata"]["long_field"] == "Y" * 50000
    assert len(data[0]["content"]) == 150001


def test_export_units_to_json_metadata_with_special_json_characters():
    """Test metadata containing special JSON characters that need escaping."""
    unit = _unit(
        "unit-special-chars",
        'Title with "quotes" and \\backslashes\\',
        'Content with newlines\nand\ttabs\tand\rcarriage returns',
        metadata={
            "quote_field": 'Value with "double quotes" and \'single quotes\'',
            "backslash_field": "Path\\to\\file\\name.txt",
            "control_chars": "Line1\nLine2\tTabbed\rReturn",
            "unicode_escape": "Unicode: \u0041\u0042\u0043",
            "forward_slash": "/path/to/resource",
            "mixed": 'Mix: "quotes", \\backslash, \n newline, \t tab',
        },
    )
    json_str = export_units_to_json([unit], mode="full")

    # Should be valid JSON despite special characters
    data = json.loads(json_str)

    assert data[0]["title"] == 'Title with "quotes" and \\backslashes\\'
    assert data[0]["content"] == 'Content with newlines\nand\ttabs\tand\rcarriage returns'

    metadata = data[0]["metadata"]
    assert metadata["quote_field"] == 'Value with "double quotes" and \'single quotes\''
    assert metadata["backslash_field"] == "Path\\to\\file\\name.txt"
    assert metadata["control_chars"] == "Line1\nLine2\tTabbed\rReturn"
    assert metadata["unicode_escape"] == "Unicode: ABC"
    assert metadata["forward_slash"] == "/path/to/resource"
    assert '\n' in metadata["mixed"]
    assert '\t' in metadata["mixed"]


def test_export_units_to_json_mixed_content_types():
    """Test collections with units of different ContentType enum values."""
    units = [
        _unit("unit-1", "Insight Unit"),
        _unit("unit-2", "Finding Unit"),
        _unit("unit-3", "Idea Unit"),
        _unit("unit-4", "Artifact Unit"),
    ]

    # Override content_type to test different enum values
    units[0].content_type = ContentType.INSIGHT
    units[1].content_type = ContentType.FINDING
    units[2].content_type = ContentType.IDEA
    units[3].content_type = ContentType.ARTIFACT

    json_str = export_units_to_json(units, mode="full")
    data = json.loads(json_str)

    assert len(data) == 4
    assert data[0]["content_type"] == "insight"
    assert data[1]["content_type"] == "finding"
    assert data[2]["content_type"] == "idea"
    assert data[3]["content_type"] == "artifact"


def test_export_units_to_json_mixed_source_projects():
    """Test collections with units from different SourceProject enum values."""
    units = [
        _unit("unit-1", "Max Unit"),
        _unit("unit-2", "Presence Unit"),
        _unit("unit-3", "Kindle Unit"),
    ]

    units[0].source_project = SourceProject.MAX
    units[1].source_project = SourceProject.PRESENCE
    units[2].source_project = SourceProject.KINDLE

    json_str = export_units_to_json(units, mode="full")
    data = json.loads(json_str)

    assert len(data) == 3
    assert data[0]["source_project"] == "max"
    assert data[1]["source_project"] == "presence"
    assert data[2]["source_project"] == "kindle"


def test_export_units_to_json_metadata_with_numeric_string_keys():
    """Test metadata with numeric string keys to ensure proper serialization."""
    unit = _unit(
        "unit-numeric-keys",
        "Numeric Keys",
        metadata={
            "123": "numeric string key",
            "456": {"nested": "value"},
            "789": [1, 2, 3],
            "normal_key": "normal value",
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert metadata["123"] == "numeric string key"
    assert metadata["456"]["nested"] == "value"
    assert metadata["789"] == [1, 2, 3]
    assert metadata["normal_key"] == "normal value"


def test_export_units_to_json_metadata_with_empty_strings():
    """Test metadata containing empty string values."""
    unit = _unit(
        "unit-empty-strings",
        "Empty String Metadata",
        metadata={
            "empty": "",
            "whitespace": "   ",
            "normal": "value",
            "nested": {"also_empty": "", "has_value": "test"},
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert metadata["empty"] == ""
    assert metadata["whitespace"] == "   "
    assert metadata["normal"] == "value"
    assert metadata["nested"]["also_empty"] == ""
    assert metadata["nested"]["has_value"] == "test"


def test_export_units_to_json_metadata_with_boolean_values():
    """Test metadata with boolean values to ensure they're not converted to strings."""
    unit = _unit(
        "unit-booleans",
        "Boolean Metadata",
        metadata={
            "is_active": True,
            "is_deleted": False,
            "flags": [True, False, True],
            "nested": {"enabled": True, "verified": False},
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert metadata["is_active"] is True
    assert metadata["is_deleted"] is False
    assert metadata["flags"] == [True, False, True]
    assert metadata["nested"]["enabled"] is True
    assert metadata["nested"]["verified"] is False


def test_export_units_to_json_metadata_with_zero_values():
    """Test metadata with zero and negative numeric values."""
    unit = _unit(
        "unit-zeros",
        "Zero Values",
        metadata={
            "zero_int": 0,
            "zero_float": 0.0,
            "negative_int": -42,
            "negative_float": -3.14,
            "very_small": 0.00000001,
            "very_large": 999999999999,
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert metadata["zero_int"] == 0
    assert metadata["zero_float"] == 0.0
    assert metadata["negative_int"] == -42
    assert metadata["negative_float"] == -3.14
    assert metadata["very_small"] == 0.00000001
    assert metadata["very_large"] == 999999999999


def test_export_units_to_json_empty_title_and_content():
    """Test units with empty strings for title and content."""
    # Create unit directly to avoid _unit helper's default content
    unit = KnowledgeUnit(
        id="unit-empty-fields",
        source_project=SourceProject.MAX,
        source_id="source-unit-empty-fields",
        source_entity_type="note",
        title="",
        content="",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
        confidence=0.8,
        utility_score=0.6,
        created_at=UNIT_TIME,
        ingested_at=UNIT_TIME,
        updated_at=UPDATED_TIME,
    )

    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    assert data[0]["title"] == ""
    assert data[0]["content"] == ""

    # Compact mode should also handle empty strings
    json_str_compact = export_units_to_json([unit], mode="compact")
    data_compact = json.loads(json_str_compact)

    assert data_compact[0]["title"] == ""
    assert data_compact[0]["content"] == ""


def test_export_units_to_json_whitespace_only_fields():
    """Test units with whitespace-only title and content."""
    unit = _unit("unit-whitespace", "   ", "\n\t  \r\n")

    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    assert data[0]["title"] == "   "
    assert data[0]["content"] == "\n\t  \r\n"


def test_export_units_to_json_metadata_with_list_of_dicts():
    """Test metadata containing lists of dictionaries."""
    unit = _unit(
        "unit-list-dicts",
        "List of Dicts",
        metadata={
            "items": [
                {"id": 1, "name": "first", "active": True},
                {"id": 2, "name": "second", "active": False},
                {"id": 3, "name": "third", "active": True},
            ],
            "nested_lists": [[{"a": 1}, {"b": 2}], [{"c": 3}]],
        },
    )
    json_str = export_units_to_json([unit], mode="full")
    data = json.loads(json_str)

    metadata = data[0]["metadata"]
    assert len(metadata["items"]) == 3
    assert metadata["items"][0] == {"id": 1, "name": "first", "active": True}
    assert metadata["items"][1] == {"id": 2, "name": "second", "active": False}
    assert metadata["items"][2] == {"id": 3, "name": "third", "active": True}
    assert metadata["nested_lists"] == [[{"a": 1}, {"b": 2}], [{"c": 3}]]


def test_export_units_to_json_large_collection():
    """Test exporting a large collection of units."""
    units = [_unit(f"unit-{i}", f"Title {i}", f"Content {i}") for i in range(1000)]

    json_str = export_units_to_json(units, mode="compact")
    data = json.loads(json_str)

    assert len(data) == 1000
    assert data[0]["id"] == "unit-0"
    assert data[999]["id"] == "unit-999"


def test_export_units_to_json_compact_mode_excludes_metadata_and_tags():
    """Test that compact mode truly excludes all fields except id, title, content."""
    unit = _unit(
        "unit-compact-test",
        "Compact Test",
        "Compact content",
        tags=["tag1", "tag2"],
        metadata={"key": "value", "nested": {"data": "here"}},
        confidence=0.95,
        utility_score=0.85,
    )

    json_str = export_units_to_json([unit], mode="compact")
    data = json.loads(json_str)

    # Should have exactly 3 keys
    assert set(data[0].keys()) == {"id", "title", "content"}
    # Should not have these fields
    assert "tags" not in data[0]
    assert "metadata" not in data[0]
    assert "confidence" not in data[0]
    assert "utility_score" not in data[0]
    assert "created_at" not in data[0]
