"""Tests for Markdown export adapter."""

from __future__ import annotations

from datetime import datetime, timezone

from graph.exports import export_units_to_markdown, get_exporter, list_exporters
from graph.exports.registry import export_units
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


# --- Basic structure generation ---


def test_export_generates_title():
    """Test that export generates document title."""
    md = export_units_to_markdown([_unit("u1", "Test")])
    assert md.startswith("# Knowledge Graph Export\n")


def test_export_custom_title():
    """Test custom document title."""
    md = export_units_to_markdown([_unit("u1", "Test")], title="My Export")
    assert "# My Export\n" in md


def test_export_groups_units_by_source_project():
    """Test units are grouped under source_project headings."""
    units = [
        _unit("u1", "Unit A", source_project=SourceProject.MAX),
        _unit("u2", "Unit B", source_project=SourceProject.KINDLE),
    ]
    md = export_units_to_markdown(units)

    assert "## max\n" in md
    assert "## kindle\n" in md


def test_export_renders_unit_title_as_h4():
    """Test that unit titles are rendered as h4 headings."""
    md = export_units_to_markdown([_unit("u1", "My Unit Title")])
    assert "#### My Unit Title\n" in md


def test_export_renders_unit_content():
    """Test that unit content is included."""
    md = export_units_to_markdown([_unit("u1", "Title", "The actual content")])
    assert "The actual content\n" in md


# --- TOC generation ---


def test_export_includes_toc():
    """Test that table of contents is generated."""
    units = [
        _unit("u1", "A", source_project=SourceProject.MAX, tags=["alpha"]),
        _unit("u2", "B", source_project=SourceProject.KINDLE, tags=["beta"]),
    ]
    md = export_units_to_markdown(units)

    assert "## Table of Contents\n" in md
    assert "- [max](#max)" in md
    assert "- [kindle](#kindle)" in md


def test_export_toc_includes_tag_subentries():
    """Test that TOC includes tag sub-entries."""
    units = [
        _unit("u1", "A", source_project=SourceProject.MAX, tags=["research"]),
    ]
    md = export_units_to_markdown(units)

    assert "  - [research](#max-research)" in md


def test_export_toc_can_be_disabled():
    """Test that TOC generation can be disabled."""
    md = export_units_to_markdown([_unit("u1", "Test")], include_toc=False)
    assert "## Table of Contents" not in md


# --- Metadata table formatting ---


def test_export_renders_metadata_table():
    """Test that metadata is rendered as a table."""
    unit = _unit("u1", "Title", metadata={"author": "Jane", "version": "1.0"})
    md = export_units_to_markdown([unit])

    assert "| Field | Value |" in md
    assert "|-------|-------|" in md
    assert "| author | Jane |" in md
    assert "| version | 1.0 |" in md


def test_export_metadata_table_can_be_disabled():
    """Test that metadata table can be disabled."""
    unit = _unit("u1", "Title", metadata={"key": "val"})
    md = export_units_to_markdown([unit], include_metadata=False)

    assert "| Field | Value |" not in md


def test_export_metadata_sorts_keys():
    """Test that metadata keys are sorted alphabetically."""
    unit = _unit("u1", "Title", metadata={"zebra": "z", "apple": "a"})
    md = export_units_to_markdown([unit])

    apple_pos = md.find("| apple |")
    zebra_pos = md.find("| zebra |")
    assert apple_pos < zebra_pos


# --- Edge relationship rendering ---


def test_export_renders_edges_as_mermaid():
    """Test that edges are rendered as Mermaid diagram."""
    units = [
        _unit("u1", "Node A"),
        _unit("u2", "Node B"),
    ]
    edges = [("u1", "u2", "builds_on")]
    md = export_units_to_markdown(units, include_edges=True, edges=edges)

    assert "## Relationships\n" in md
    assert "```mermaid" in md
    assert "graph LR" in md
    assert "builds_on" in md
    assert "```" in md


def test_export_edges_not_rendered_without_flag():
    """Test that edges are not rendered when include_edges=False."""
    units = [_unit("u1", "A"), _unit("u2", "B")]
    edges = [("u1", "u2", "relates_to")]
    md = export_units_to_markdown(units, include_edges=False, edges=edges)

    assert "## Relationships" not in md
    assert "```mermaid" not in md


def test_export_edges_not_rendered_when_empty():
    """Test that relationships section is omitted when edges list is empty."""
    md = export_units_to_markdown([_unit("u1", "A")], include_edges=True, edges=[])
    assert "## Relationships" not in md


# --- Special character escaping ---


def test_export_escapes_markdown_special_chars_in_title():
    """Test that Markdown special characters are escaped in titles."""
    unit = _unit("u1", "Title *with* _special_ `chars`")
    md = export_units_to_markdown([unit])

    # The unit title heading should have escaped chars
    assert "\\*with\\*" in md
    assert "\\_special\\_" in md
    assert "\\`chars\\`" in md


def test_export_escapes_pipe_in_metadata_table():
    """Test that pipes are escaped in metadata table cells."""
    unit = _unit("u1", "Title", metadata={"key": "value|with|pipes"})
    md = export_units_to_markdown([unit])

    assert "value\\|with\\|pipes" in md


def test_export_handles_newlines_in_metadata():
    """Test that newlines in metadata values are replaced for table."""
    unit = _unit("u1", "Title", metadata={"key": "line1\nline2\nline3"})
    md = export_units_to_markdown([unit])

    # In table cells, newlines should be replaced with spaces
    assert "line1 line2 line3" in md


# --- Empty graph ---


def test_export_empty_units():
    """Test export of empty unit list."""
    md = export_units_to_markdown([])

    assert "# Knowledge Graph Export\n" in md
    assert "*No knowledge units to export.*" in md


# --- Single unit ---


def test_export_single_unit_complete():
    """Test complete rendering of a single unit."""
    unit = _unit(
        "u1",
        "Single Unit",
        "Content here",
        tags=["tag1", "tag2"],
        metadata={"source": "test"},
    )
    md = export_units_to_markdown([unit])

    assert "#### Single Unit\n" in md
    assert "Content here\n" in md
    assert "| source | test |" in md
    assert "`tag1`" in md
    assert "`tag2`" in md


# --- Large graphs ---


def test_export_large_number_of_units():
    """Test export of many units."""
    units = [_unit(f"u-{i}", f"Unit {i}", tags=[f"group-{i % 5}"]) for i in range(100)]
    md = export_units_to_markdown(units)

    assert md.count("####") == 100
    assert "Unit 0" in md
    assert "Unit 99" in md


# --- Units without tags ---


def test_export_units_without_tags_go_to_untagged_section():
    """Test that units without tags are placed in 'Untagged' section."""
    unit = _unit("u1", "No Tags", tags=[])
    md = export_units_to_markdown([unit])

    assert "### Untagged\n" in md


def test_export_mix_of_tagged_and_untagged():
    """Test mix of tagged and untagged units."""
    units = [
        _unit("u1", "Tagged", tags=["alpha"]),
        _unit("u2", "Untagged", tags=[]),
    ]
    md = export_units_to_markdown(units)

    assert "### alpha\n" in md
    assert "### Untagged\n" in md


# --- Markdown syntax validation ---


def test_export_valid_markdown_headings():
    """Test that headings follow proper Markdown syntax."""
    units = [
        _unit("u1", "A", source_project=SourceProject.MAX, tags=["t1"]),
    ]
    md = export_units_to_markdown(units)

    lines = md.split("\n")
    for line in lines:
        if line.startswith("#"):
            # Heading must have space after # characters
            stripped = line.lstrip("#")
            assert stripped.startswith(" ") or stripped == ""


def test_export_valid_markdown_table():
    """Test that tables follow proper Markdown table syntax."""
    unit = _unit("u1", "Title", metadata={"k": "v"})
    md = export_units_to_markdown([unit])

    lines = md.split("\n")
    table_lines = [l for l in lines if l.startswith("|")]
    assert len(table_lines) >= 3  # header + separator + at least one row
    # Separator line should be all dashes and pipes
    sep = table_lines[1]
    assert all(c in "|-: " for c in sep)


def test_export_renders_tags_as_code_spans():
    """Test that tags are rendered as inline code."""
    unit = _unit("u1", "Title", tags=["my-tag"])
    md = export_units_to_markdown([unit])

    assert "`my-tag`" in md


def test_export_renders_timestamps():
    """Test that timestamps are rendered in italic."""
    unit = _unit("u1", "Title", created_at=UNIT_TIME, updated_at=UPDATED_TIME)
    md = export_units_to_markdown([unit])

    assert "*Created: 2026-05-01 10:15:30 UTC" in md
    assert "Updated: 2026-05-02 08:30:45 UTC" in md


# --- Registry integration ---


def test_markdown_exporter_registered_in_registry():
    """Test that Markdown exporter is registered in the export registry."""
    exporters = list_exporters()
    assert "markdown" in exporters


def test_get_markdown_exporter_from_registry():
    """Test retrieving Markdown exporter from registry."""
    exporter = get_exporter("markdown")
    assert exporter is export_units_to_markdown


def test_markdown_exporter_works_through_registry():
    """Test that Markdown export works when called through registry."""
    units = [_unit("u1", "Registry Test")]
    md = export_units("markdown", units, title="Via Registry")

    assert "# Via Registry\n" in md
    assert "Registry Test" in md
