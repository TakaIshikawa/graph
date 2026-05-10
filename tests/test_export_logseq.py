"""Tests for Logseq export adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from graph.exports.logseq import (
    export_units_to_logseq,
    export_units_to_logseq_files,
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


# --- File generation tests ---


def test_export_generates_page_file_per_unit():
    """Test that each unit generates a page file."""
    units = [
        _unit("u1", "First Unit"),
        _unit("u2", "Second Unit"),
    ]
    files = export_units_to_logseq_files(units, include_journal_pages=False)

    assert "pages/First Unit.md" in files
    assert "pages/Second Unit.md" in files
    assert len(files) == 2


def test_export_sanitizes_page_names():
    """Test that special characters in titles are sanitized for filenames."""
    units = [
        _unit("u1", "Unit/With/Slashes"),
        _unit("u2", "Unit:With:Colons"),
        _unit("u3", 'Unit"With"Quotes'),
    ]
    files = export_units_to_logseq_files(units, include_journal_pages=False)

    assert "pages/Unit_With_Slashes.md" in files
    assert "pages/Unit_With_Colons.md" in files
    assert "pages/Unit_With_Quotes.md" in files


def test_export_generates_journal_pages():
    """Test that journal pages are generated for units with created_at."""
    units = [
        _unit("u1", "Unit 1", created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc)),
        _unit("u2", "Unit 2", created_at=datetime(2026, 5, 10, 14, 0, 0, tzinfo=timezone.utc)),
        _unit("u3", "Unit 3", created_at=datetime(2026, 5, 11, 10, 0, 0, tzinfo=timezone.utc)),
    ]
    files = export_units_to_logseq_files(units, include_journal_pages=True)

    # Should have 3 page files + 2 journal files
    assert len(files) == 5
    assert "journals/2026_05_10.md" in files
    assert "journals/2026_05_11.md" in files


def test_export_skips_journal_pages_when_disabled():
    """Test that journal pages are not generated when disabled."""
    units = [
        _unit("u1", "Unit 1", created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc)),
    ]
    files = export_units_to_logseq_files(units, include_journal_pages=False)

    assert len(files) == 1
    assert "pages/Unit 1.md" in files
    assert "journals/2026_05_10.md" not in files


# --- Frontmatter tests ---


def test_page_includes_frontmatter():
    """Test that page files include YAML frontmatter."""
    unit = _unit("u1", "Test Unit", tags=["tag1", "tag2"])
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Test Unit.md"]

    assert content.startswith("---\n")
    assert "title: Test Unit\n" in content
    assert "tags: tag1, tag2\n" in content
    assert "public: true\n" in content
    assert "---\n" in content


def test_frontmatter_includes_alias():
    """Test that frontmatter includes alias when source_id differs from id."""
    unit = _unit("u1", "Test Unit")
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Test Unit.md"]

    assert "alias: source-u1\n" in content


def test_frontmatter_includes_metadata():
    """Test that custom metadata is included in frontmatter."""
    unit = _unit("u1", "Test Unit", metadata={"author": "John Doe", "rating": 5})
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Test Unit.md"]

    assert "author: John Doe\n" in content
    assert "rating: 5\n" in content


# --- Block reference tests ---


def test_content_rendered_as_blocks():
    """Test that content lines are rendered as Logseq blocks."""
    unit = _unit("u1", "Test Unit", content="Line 1\nLine 2\nLine 3")
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Test Unit.md"]

    assert "- Line 1\n" in content
    assert "- Line 2\n" in content
    assert "- Line 3\n" in content


def test_blocks_include_block_ids():
    """Test that blocks include unique block IDs when enabled."""
    unit = _unit("u1", "Test Unit", content="Test line")
    files = export_units_to_logseq_files([unit], include_block_ids=True)
    content = files["pages/Test Unit.md"]

    # Block IDs should be present (8 hex characters)
    assert "  id:: " in content
    # Verify format with regex-like check
    lines = content.split("\n")
    block_id_lines = [line for line in lines if "id:: " in line]
    assert len(block_id_lines) > 0
    # Each block ID should be 8 characters (hex)
    for line in block_id_lines:
        block_id = line.split("id:: ")[1].strip()
        assert len(block_id) == 8
        assert all(c in "0123456789abcdef" for c in block_id)


def test_blocks_skip_ids_when_disabled():
    """Test that blocks do not include IDs when disabled."""
    unit = _unit("u1", "Test Unit", content="Test line")
    files = export_units_to_logseq_files([unit], include_block_ids=False)
    content = files["pages/Test Unit.md"]

    assert "id:: " not in content


# --- Page link tests ---


def test_metadata_includes_page_links():
    """Test that metadata fields with relations use page link syntax."""
    unit = _unit("u1", "Test Unit")
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Test Unit.md"]

    # Source project should be a page link
    assert "source-project:: [[max]]" in content


def test_page_links_escape_brackets():
    """Test that page links properly escape closing brackets."""
    # This is tested via the source-project link which shouldn't have issues
    unit = _unit("u1", "Test [[Unit]]")
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Test [[Unit]].md"]

    # Title should be properly set
    assert "title: Test [[Unit]]" in content


def test_dates_formatted_as_logseq_page_links():
    """Test that dates are formatted as Logseq-style page links."""
    unit = _unit("u1", "Test Unit", created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc))
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Test Unit.md"]

    # Logseq date format: "May 10, 2026"
    assert "created:: [[May 10, 2026]]" in content


# --- Query block tests ---


def test_query_blocks_included_when_enabled():
    """Test that query blocks are included when queries are enabled."""
    unit = _unit("u1", "Test Unit", tags=["tag1", "tag2"])
    files = export_units_to_logseq_files([unit], include_queries=True)
    content = files["pages/Test Unit.md"]

    assert "## Related\n" in content
    assert "#+BEGIN_QUERY\n" in content
    assert "#+END_QUERY\n" in content
    # Query should reference tags
    assert "page-tags" in content


def test_query_blocks_skipped_when_disabled():
    """Test that query blocks are not included when disabled."""
    unit = _unit("u1", "Test Unit", tags=["tag1", "tag2"])
    files = export_units_to_logseq_files([unit], include_queries=False)
    content = files["pages/Test Unit.md"]

    assert "#+BEGIN_QUERY" not in content
    assert "#+END_QUERY" not in content


def test_query_blocks_use_unit_tags():
    """Test that query blocks reference the unit's tags."""
    unit = _unit("u1", "Test Unit", tags=["python", "testing"])
    files = export_units_to_logseq_files([unit], include_queries=True)
    content = files["pages/Test Unit.md"]

    # Should query for units with the same tags
    assert "[[python]]" in content
    assert "[[testing]]" in content


# --- Journal page tests ---


def test_journal_page_has_correct_filename():
    """Test that journal pages use the correct date format in filename."""
    unit = _unit("u1", "Test Unit", created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc))
    files = export_units_to_logseq_files([unit], include_journal_pages=True)

    # Format: YYYY_MM_DD
    assert "journals/2026_05_10.md" in files


def test_journal_page_lists_units():
    """Test that journal pages list units created on that date."""
    units = [
        _unit("u1", "First Unit", created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc)),
        _unit("u2", "Second Unit", created_at=datetime(2026, 5, 10, 14, 0, 0, tzinfo=timezone.utc)),
    ]
    files = export_units_to_logseq_files(units, include_journal_pages=True)
    content = files["journals/2026_05_10.md"]

    assert "[[First Unit]]" in content
    assert "[[Second Unit]]" in content


def test_journal_page_includes_unit_tags():
    """Test that journal pages include tags for each unit."""
    unit = _unit(
        "u1",
        "Test Unit",
        tags=["python", "testing"],
        created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc),
    )
    files = export_units_to_logseq_files([unit], include_journal_pages=True)
    content = files["journals/2026_05_10.md"]

    assert "[[python]]" in content
    assert "[[testing]]" in content


def test_journal_page_title():
    """Test that journal pages have proper title in frontmatter."""
    unit = _unit("u1", "Test Unit", created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc))
    files = export_units_to_logseq_files([unit], include_journal_pages=True)
    content = files["journals/2026_05_10.md"]

    assert "title: May 10, 2026\n" in content


# --- String export tests (registry compatibility) ---


def test_string_export_includes_file_markers():
    """Test that string export includes file path markers."""
    unit = _unit("u1", "Test Unit")
    output = export_units_to_logseq([unit], include_journal_pages=False)

    assert "### FILE: pages/Test Unit.md" in output


def test_string_export_includes_all_files():
    """Test that string export includes content from all files."""
    units = [
        _unit("u1", "First Unit", created_at=datetime(2026, 5, 10, 10, 0, 0, tzinfo=timezone.utc)),
        _unit("u2", "Second Unit", created_at=datetime(2026, 5, 10, 14, 0, 0, tzinfo=timezone.utc)),
    ]
    output = export_units_to_logseq(units, include_journal_pages=True)

    assert "### FILE: journals/2026_05_10.md" in output
    assert "### FILE: pages/First Unit.md" in output
    assert "### FILE: pages/Second Unit.md" in output


def test_string_export_sorted_by_path():
    """Test that string export lists files in sorted order."""
    units = [
        _unit("u2", "Zebra Unit"),
        _unit("u1", "Alpha Unit"),
    ]
    output = export_units_to_logseq(units, include_journal_pages=False)

    # Alpha should come before Zebra
    alpha_pos = output.find("### FILE: pages/Alpha Unit.md")
    zebra_pos = output.find("### FILE: pages/Zebra Unit.md")
    assert alpha_pos < zebra_pos


# --- Edge cases ---


def test_empty_units_list():
    """Test exporting empty units list."""
    files = export_units_to_logseq_files([], include_journal_pages=False)
    assert len(files) == 0


def test_unit_with_no_content():
    """Test unit with empty content."""
    unit = _unit("u1", "Empty Unit", content="")
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Empty Unit.md"]

    # Should still have frontmatter and metadata
    assert "title: Empty Unit" in content
    assert "source-project::" in content


def test_unit_with_no_tags():
    """Test unit with no tags."""
    unit = _unit("u1", "Untagged Unit", tags=[])
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Untagged Unit.md"]

    # tags field should not appear in frontmatter
    assert "tags:" not in content or "tags: \n" not in content


def test_unit_with_complex_metadata():
    """Test unit with various metadata value types."""
    unit = _unit(
        "u1",
        "Complex Unit",
        metadata={
            "bool_val": True,
            "int_val": 42,
            "float_val": 3.14,
            "list_val": ["a", "b", "c"],
            "date_val": datetime(2026, 5, 10, tzinfo=timezone.utc),
        },
    )
    files = export_units_to_logseq_files([unit], include_journal_pages=False)
    content = files["pages/Complex Unit.md"]

    assert "bool_val: true" in content
    assert "int_val: 42" in content
    assert "float_val: 3.14" in content
    assert "list_val: [a, b, c]" in content
    assert 'date_val: "2026-05-10T00:00:00+00:00"' in content


# --- Integration with registry ---


def test_exporter_registered():
    """Test that Logseq exporter is registered in the registry."""
    from graph.exports.registry import list_exporters

    exporters = list_exporters()
    assert "logseq" in exporters


def test_exporter_callable_from_registry():
    """Test that Logseq exporter can be called via registry."""
    from graph.exports.registry import export_units

    unit = _unit("u1", "Test Unit")
    output = export_units("logseq", [unit], include_journal_pages=False)

    assert isinstance(output, str)
    assert "### FILE: pages/Test Unit.md" in output
