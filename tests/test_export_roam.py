"""Tests for Roam Research export adapter."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from graph.exports.roam import (
    export_units_to_roam,
    export_units_to_roam_pages,
)
from graph.types.enums import ContentType, SourceProject
from graph.types.models import KnowledgeUnit

UNIT_TIME = datetime(2026, 5, 10, 10, 15, 30, tzinfo=timezone.utc)
UPDATED_TIME = datetime(2026, 5, 11, 8, 30, 45, tzinfo=timezone.utc)


def _unit(
    unit_id: str,
    title: str,
    content: str = "",
    *,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
    source_project: SourceProject = SourceProject.MAX,
) -> KnowledgeUnit:
    """Create a test unit with sensible defaults."""
    return KnowledgeUnit(
        id=unit_id,
        source_project=source_project,
        source_id=f"source-{unit_id}",
        source_entity_type="note",
        title=title,
        content=content or f"{title} content",
        content_type=ContentType.INSIGHT,
        metadata=metadata or {},
        tags=tags or [],
        created_at=created_at or UNIT_TIME,
        updated_at=updated_at or UPDATED_TIME,
    )


# --- Page structure tests ---


def test_export_generates_page_per_unit():
    """Test that each unit generates a Roam page."""
    units = [
        _unit("u1", "First Page"),
        _unit("u2", "Second Page"),
    ]
    pages = export_units_to_roam_pages(units)

    assert len(pages) == 2
    assert pages[0]["title"] == "First Page"
    assert pages[1]["title"] == "Second Page"


def test_page_includes_title():
    """Test that pages include the unit title."""
    unit = _unit("u1", "Test Page Title")
    pages = export_units_to_roam_pages([unit])

    assert pages[0]["title"] == "Test Page Title"


def test_page_includes_children():
    """Test that pages include children blocks."""
    unit = _unit("u1", "Test Page", content="Test content")
    pages = export_units_to_roam_pages([unit])

    assert "children" in pages[0]
    assert isinstance(pages[0]["children"], list)
    assert len(pages[0]["children"]) > 0


def test_page_includes_timestamps():
    """Test that pages include create and edit timestamps."""
    unit = _unit(
        "u1",
        "Test Page",
        created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 5, 11, 10, 0, 0, tzinfo=timezone.utc),
    )
    pages = export_units_to_roam_pages([unit])

    assert "create-time" in pages[0]
    assert "edit-time" in pages[0]
    # Timestamps should be in milliseconds
    assert pages[0]["create-time"] == int(datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
    assert pages[0]["edit-time"] == int(datetime(2026, 5, 11, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)


# --- Block structure tests ---


def test_content_rendered_as_blocks():
    """Test that content lines are rendered as block children."""
    unit = _unit("u1", "Test Page", content="Line 1\nLine 2\nLine 3")
    pages = export_units_to_roam_pages([unit])

    children = pages[0]["children"]
    # Find content blocks (not attribute blocks)
    content_blocks = [child for child in children if "::" not in child["string"]]

    assert len(content_blocks) == 3
    assert content_blocks[0]["string"] == "Line 1"
    assert content_blocks[1]["string"] == "Line 2"
    assert content_blocks[2]["string"] == "Line 3"


def test_blocks_include_uids():
    """Test that blocks include UIDs when enabled."""
    unit = _unit("u1", "Test Page", content="Test line")
    pages = export_units_to_roam_pages([unit], include_block_refs=True)

    children = pages[0]["children"]
    for child in children:
        assert "uid" in child
        # UID should be 9 characters (alphanumeric)
        assert len(child["uid"]) == 9
        assert all(c in "0123456789abcdef" for c in child["uid"])


def test_blocks_skip_uids_when_disabled():
    """Test that blocks do not include UIDs when disabled."""
    unit = _unit("u1", "Test Page", content="Test line")
    pages = export_units_to_roam_pages([unit], include_block_refs=False)

    children = pages[0]["children"]
    for child in children:
        assert "uid" not in child


# --- Attribute tests ---


def test_attributes_use_double_colon_syntax():
    """Test that attributes use :key:: value syntax."""
    unit = _unit("u1", "Test Page")
    pages = export_units_to_roam_pages([unit], include_attributes=True)

    children = pages[0]["children"]
    # Find attribute blocks
    attr_blocks = [child for child in children if "::" in child["string"]]

    assert len(attr_blocks) > 0
    # Should have source-project attribute
    source_proj_block = next(
        (child for child in attr_blocks if child["string"].startswith("source-project::")), None
    )
    assert source_proj_block is not None
    assert "source-project:: [[max]]" in source_proj_block["string"]


def test_metadata_exported_as_attributes():
    """Test that custom metadata is exported as attributes."""
    unit = _unit("u1", "Test Page", metadata={"author": "John Doe", "rating": 5})
    pages = export_units_to_roam_pages([unit], include_attributes=True)

    children = pages[0]["children"]
    attr_strings = [child["string"] for child in children if "::" in child["string"]]

    assert any("author:: John Doe" in s for s in attr_strings)
    assert any("rating:: 5" in s for s in attr_strings)


def test_attributes_skipped_when_disabled():
    """Test that attributes are not included when disabled."""
    unit = _unit("u1", "Test Page", metadata={"author": "John Doe"})
    pages = export_units_to_roam_pages([unit], include_attributes=False)

    children = pages[0]["children"]
    # Should only have content blocks, no attribute blocks
    attr_blocks = [child for child in children if "::" in child["string"] and child["string"].startswith("author::")]

    assert len(attr_blocks) == 0


# --- Page reference tests ---


def test_page_references_use_double_brackets():
    """Test that page references use [[page name]] syntax."""
    unit = _unit("u1", "Test Page")
    pages = export_units_to_roam_pages([unit], include_attributes=True)

    children = pages[0]["children"]
    # Source project should be a page reference
    source_proj_block = next(
        (child for child in children if "source-project::" in child["string"]), None
    )
    assert source_proj_block is not None
    assert "[[max]]" in source_proj_block["string"]


def test_tags_exported_as_page_references():
    """Test that tags are exported as page references."""
    unit = _unit("u1", "Test Page", tags=["python", "testing"])
    pages = export_units_to_roam_pages([unit], include_attributes=True)

    children = pages[0]["children"]
    # Find tags block
    tags_block = next(
        (child for child in children if "tags::" in child["string"]), None
    )
    assert tags_block is not None
    assert "[[python]]" in tags_block["string"]
    assert "[[testing]]" in tags_block["string"]


def test_page_references_escape_brackets():
    """Test that page references properly escape closing brackets."""
    # Create a unit with brackets in a tag
    unit = _unit("u1", "Test Page", tags=["tag[[with]]brackets"])
    pages = export_units_to_roam_pages([unit], include_attributes=True)

    children = pages[0]["children"]
    tags_block = next(
        (child for child in children if "tags::" in child["string"]), None
    )
    # Should escape the closing brackets
    assert tags_block is not None
    # The escaped form should be present
    assert r"\]\]" in tags_block["string"]


# --- Date formatting tests ---


def test_dates_formatted_with_ordinal_suffix():
    """Test that dates are formatted with ordinal suffixes."""
    unit = _unit("u1", "Test Page", created_at=datetime(2026, 5, 1, 10, 0, 0, tzinfo=timezone.utc))
    pages = export_units_to_roam_pages([unit], include_attributes=True)

    children = pages[0]["children"]
    created_block = next(
        (child for child in children if "created::" in child["string"]), None
    )
    assert created_block is not None
    assert "May 1st, 2026" in created_block["string"]


def test_dates_various_ordinal_suffixes():
    """Test that different ordinal suffixes are correctly applied."""
    # 1st, 2nd, 3rd, 4th, etc.
    test_cases = [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (31, "31st"),
    ]

    for day, expected_suffix in test_cases:
        unit = _unit("u1", "Test", created_at=datetime(2026, 5, day, 10, 0, 0, tzinfo=timezone.utc))
        pages = export_units_to_roam_pages([unit], include_attributes=True)
        children = pages[0]["children"]
        created_block = next(
            (child for child in children if "created::" in child["string"]), None
        )
        assert f"May {expected_suffix}, 2026" in created_block["string"]


# --- JSON export tests ---


def test_json_export_produces_valid_json():
    """Test that JSON export produces valid parseable JSON."""
    unit = _unit("u1", "Test Page", content="Test content")
    output = export_units_to_roam([unit], format="json")

    # Should be valid JSON
    parsed = json.loads(output)
    assert isinstance(parsed, list)
    assert len(parsed) == 1


def test_json_export_includes_page_structure():
    """Test that JSON export includes full page structure."""
    unit = _unit("u1", "Test Page", content="Test content")
    output = export_units_to_roam([unit], format="json")
    parsed = json.loads(output)

    page = parsed[0]
    assert page["title"] == "Test Page"
    assert "children" in page
    assert len(page["children"]) > 0


def test_json_export_formatted_with_indentation():
    """Test that JSON export is formatted with indentation."""
    unit = _unit("u1", "Test Page")
    output = export_units_to_roam([unit], format="json")

    # Should have indentation
    assert "  " in output or "\t" in output


# --- EDN export tests ---


def test_edn_export_produces_edn_format():
    """Test that EDN export produces EDN-formatted output."""
    unit = _unit("u1", "Test Page", content="Test content")
    output = export_units_to_roam([unit], format="edn")

    # EDN uses Clojure-like syntax
    assert output.startswith("[")
    assert output.strip().endswith("]")
    assert ":title" in output
    assert ":children" in output


def test_edn_export_includes_page_title():
    """Test that EDN export includes page title."""
    unit = _unit("u1", "Test Page Title")
    output = export_units_to_roam([unit], format="edn")

    assert ':title "Test Page Title"' in output


def test_edn_export_includes_block_strings():
    """Test that EDN export includes block string content."""
    unit = _unit("u1", "Test Page", content="Test content line")
    output = export_units_to_roam([unit], format="edn")

    assert ':string "Test content line"' in output


def test_edn_export_includes_block_uids():
    """Test that EDN export includes block UIDs."""
    unit = _unit("u1", "Test Page", content="Test content")
    output = export_units_to_roam([unit], format="edn", include_block_refs=True)

    assert ":uid" in output


def test_edn_export_escapes_strings():
    """Test that EDN export properly escapes string content."""
    unit = _unit("u1", "Test Page", content='Content with "quotes" and \\backslashes')
    output = export_units_to_roam([unit], format="edn")

    # Quotes and backslashes should be escaped
    assert r'\"' in output
    assert r'\\' in output


def test_edn_export_escapes_newlines():
    """Test that EDN export escapes newlines in strings."""
    unit = _unit("u1", "Test Page", content="Line 1\nLine 2")
    output = export_units_to_roam([unit], format="edn")

    # Newlines should be escaped (though content is split into blocks, this tests the escaping)
    # Each line becomes a separate block, but if they had literal newlines within, they'd be escaped
    # Let's test with metadata that might have newlines
    unit = _unit("u1", "Test Page", metadata={"note": "Multi\nline\nvalue"})
    output = export_units_to_roam([unit], format="edn")

    # Should escape newlines
    assert "\\n" in output


# --- Format validation tests ---


def test_invalid_format_raises_error():
    """Test that invalid format raises ValueError."""
    unit = _unit("u1", "Test Page")

    try:
        export_units_to_roam([unit], format="invalid")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unsupported format" in str(e)


# --- Edge cases ---


def test_empty_units_list():
    """Test exporting empty units list."""
    pages = export_units_to_roam_pages([])
    assert len(pages) == 0

    output = export_units_to_roam([], format="json")
    parsed = json.loads(output)
    assert len(parsed) == 0


def test_unit_with_no_content():
    """Test unit with empty content."""
    # Create unit with truly empty content (bypassing the helper's default)
    unit = KnowledgeUnit(
        id="u1",
        source_project=SourceProject.MAX,
        source_id="source-u1",
        source_entity_type="note",
        title="Empty Page",
        content="",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
        created_at=UNIT_TIME,
        updated_at=UPDATED_TIME,
    )
    pages = export_units_to_roam_pages([unit])

    # Should still have a page with attributes
    assert len(pages) == 1
    assert pages[0]["title"] == "Empty Page"
    # Should have attribute blocks but no content blocks
    children = pages[0]["children"]
    content_blocks = [child for child in children if "::" not in child["string"]]
    assert len(content_blocks) == 0


def test_unit_with_no_tags():
    """Test unit with no tags."""
    unit = _unit("u1", "Untagged Page", tags=[])
    pages = export_units_to_roam_pages([unit], include_attributes=True)

    children = pages[0]["children"]
    # Should not have a tags block
    tags_blocks = [child for child in children if "tags::" in child["string"]]
    assert len(tags_blocks) == 0


def test_unit_with_complex_metadata():
    """Test unit with various metadata value types."""
    unit = _unit(
        "u1",
        "Complex Page",
        metadata={
            "bool_val": True,
            "int_val": 42,
            "float_val": 3.14,
            "list_val": ["a", "b", "c"],
            "date_val": datetime(2026, 5, 10, tzinfo=timezone.utc),
        },
    )
    pages = export_units_to_roam_pages([unit], include_attributes=True)

    children = pages[0]["children"]
    attr_strings = [child["string"] for child in children if "::" in child["string"]]

    assert any("bool_val:: true" in s for s in attr_strings)
    assert any("int_val:: 42" in s for s in attr_strings)
    assert any("float_val:: 3.14" in s for s in attr_strings)
    assert any("list_val:: a, b, c" in s for s in attr_strings)
    assert any("May 10th, 2026" in s for s in attr_strings)


# --- Integration with registry ---


def test_exporter_registered():
    """Test that Roam exporter is registered in the registry."""
    from graph.exports.registry import list_exporters

    exporters = list_exporters()
    assert "roam" in exporters


def test_exporter_callable_from_registry():
    """Test that Roam exporter can be called via registry."""
    from graph.exports.registry import export_units

    unit = _unit("u1", "Test Page")
    output = export_units("roam", [unit], format="json")

    assert isinstance(output, str)
    parsed = json.loads(output)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Test Page"
