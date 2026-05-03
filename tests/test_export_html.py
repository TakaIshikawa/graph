"""Tests for HTML export adapter."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from graph.exports import export_units_to_html, get_exporter, list_exporters
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
        created_at=created_at or UNIT_TIME,
        updated_at=updated_at or UPDATED_TIME,
    )


def test_export_units_to_html_generates_valid_html5():
    """Test that HTML export generates valid HTML5 structure."""
    units = [_unit("unit-1", "Test Unit")]
    html = export_units_to_html(units)

    assert html.startswith("<!DOCTYPE html>")
    assert '<html lang="en">' in html
    assert '<meta charset="UTF-8">' in html
    assert '<meta name="viewport"' in html
    assert "</html>" in html
    assert "<body>" in html
    assert "</body>" in html


def test_export_units_to_html_includes_title():
    """Test that custom title appears in document."""
    units = [_unit("unit-1", "Test Unit")]
    html = export_units_to_html(units, title="My Custom Export")

    assert "<title>My Custom Export</title>" in html
    assert "<h1>My Custom Export</h1>" in html


def test_export_units_to_html_uses_default_title():
    """Test default title when none provided."""
    units = [_unit("unit-1", "Test Unit")]
    html = export_units_to_html(units)

    assert "<title>Knowledge Units Export</title>" in html
    assert "<h1>Knowledge Units Export</h1>" in html


def test_export_units_to_html_renders_single_unit():
    """Test rendering a single unit with all fields."""
    unit = _unit(
        "unit-alpha",
        "Alpha Unit",
        "This is the content of alpha unit.",
        tags=["ai", "research"],
        metadata={"author": "Test Author", "version": "1.0"},
    )
    html = export_units_to_html([unit])

    assert '<article class="unit">' in html
    assert '<h2 class="unit-title">Alpha Unit</h2>' in html
    assert '<pre class="unit-content">This is the content of alpha unit.</pre>' in html
    assert '<span class="tag-badge">ai</span>' in html
    assert '<span class="tag-badge">research</span>' in html
    assert "<dt>author</dt>" in html
    assert "<dd>Test Author</dd>" in html
    assert "<dt>version</dt>" in html
    assert "<dd>1.0</dd>" in html


def test_export_units_to_html_renders_multiple_units():
    """Test rendering multiple units in sequence."""
    units = [
        _unit("unit-1", "First Unit", "First content"),
        _unit("unit-2", "Second Unit", "Second content"),
        _unit("unit-3", "Third Unit", "Third content"),
    ]
    html = export_units_to_html(units)

    assert html.count('<article class="unit">') == 3
    assert "First Unit" in html
    assert "Second Unit" in html
    assert "Third Unit" in html
    assert "First content" in html
    assert "Second content" in html
    assert "Third content" in html


def test_export_units_to_html_handles_unit_without_metadata():
    """Test unit rendering without metadata."""
    unit = _unit("unit-1", "No Metadata Unit", metadata={})
    html = export_units_to_html([unit])

    assert "No Metadata Unit" in html
    assert '<dl class="unit-metadata">' not in html


def test_export_units_to_html_handles_unit_without_tags():
    """Test unit rendering without tags."""
    unit = _unit("unit-1", "No Tags Unit", tags=[])
    html = export_units_to_html([unit])

    assert "No Tags Unit" in html
    assert '<div class="unit-tags">' not in html
    assert '<span class="tag-badge">' not in html


def test_export_units_to_html_handles_unit_without_content():
    """Test unit rendering without content."""
    unit = _unit("unit-1", "Empty Content", content="")
    html = export_units_to_html([unit])

    assert "Empty Content" in html
    assert '<pre class="unit-content"></pre>' not in html


def test_export_units_to_html_handles_empty_unit_list():
    """Test export with no units."""
    html = export_units_to_html([])

    assert "<!DOCTYPE html>" in html
    assert "<h1>Knowledge Units Export</h1>" in html
    assert '<article class="unit">' not in html


def test_export_units_to_html_escapes_html_special_characters():
    """Test that HTML special characters are properly escaped."""
    unit = _unit(
        "unit-1",
        "Title with <script>alert('xss')</script>",
        "Content with & < > \" characters",
        tags=["<tag>", "tag&special"],
        metadata={"key<>": "value&quot;"},
    )
    html = export_units_to_html([unit])

    # Title should be escaped
    assert "&lt;script&gt;" in html
    assert "<script>" not in html

    # Content should be escaped
    assert "Content with &amp; &lt; &gt; &quot; characters" in html

    # Tags should be escaped
    assert "&lt;tag&gt;" in html
    assert "tag&amp;special" in html

    # Metadata should be escaped
    assert "key&lt;&gt;" in html
    assert "value&amp;quot;" in html


def test_export_units_to_html_formats_timestamps():
    """Test timestamp formatting in HTML."""
    unit = _unit(
        "unit-1",
        "Timestamp Test",
        created_at=UNIT_TIME,
        updated_at=UPDATED_TIME,
    )
    html = export_units_to_html([unit])

    assert "Created: 2026-05-01 10:15:30" in html
    assert "Updated: 2026-05-02 08:30:45" in html


def test_export_units_to_html_sorts_tags_alphabetically():
    """Test that tags are sorted alphabetically."""
    unit = _unit("unit-1", "Tag Test", tags=["zeta", "alpha", "beta"])
    html = export_units_to_html([unit])

    # Find positions of tags in HTML
    alpha_pos = html.find('tag-badge">alpha</span>')
    beta_pos = html.find('tag-badge">beta</span>')
    zeta_pos = html.find('tag-badge">zeta</span>')

    assert alpha_pos < beta_pos < zeta_pos


def test_export_units_to_html_sorts_metadata_keys_alphabetically():
    """Test that metadata keys are sorted alphabetically."""
    unit = _unit(
        "unit-1",
        "Metadata Test",
        metadata={"zebra": "z", "apple": "a", "banana": "b"},
    )
    html = export_units_to_html([unit])

    # Find positions of metadata keys in HTML
    apple_pos = html.find("<dt>apple</dt>")
    banana_pos = html.find("<dt>banana</dt>")
    zebra_pos = html.find("<dt>zebra</dt>")

    assert apple_pos < banana_pos < zebra_pos


def test_export_units_to_html_custom_css():
    """Test that custom CSS can be provided."""
    custom_css = "body { background: red; }"
    units = [_unit("unit-1", "Custom CSS Test")]
    html = export_units_to_html(units, css=custom_css)

    assert custom_css in html
    assert "background: red;" in html


def test_export_units_to_html_uses_default_css_when_none_provided():
    """Test that default CSS is used when none provided."""
    units = [_unit("unit-1", "Default CSS Test")]
    html = export_units_to_html(units)

    assert "<style>" in html
    assert "font-family:" in html
    assert ".unit {" in html
    assert ".tag-badge {" in html


def test_export_units_to_html_handles_untitled_unit():
    """Test that units without titles show 'Untitled'."""
    unit = _unit("unit-1", "")
    html = export_units_to_html([unit])

    assert "<h2 class=\"unit-title\">Untitled</h2>" in html


def test_export_units_to_html_handles_complex_metadata():
    """Test rendering of complex metadata types."""
    unit = _unit(
        "unit-1",
        "Complex Metadata",
        metadata={
            "string": "text",
            "number": 42,
            "list": ["a", "b", "c"],
            "dict": {"nested": "value"},
            "datetime": datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc),
        },
    )
    html = export_units_to_html([unit])

    assert "<dt>string</dt>" in html
    assert "<dd>text</dd>" in html
    assert "<dt>number</dt>" in html
    assert "<dd>42</dd>" in html
    assert "<dt>list</dt>" in html
    assert "<dd>a, b, c</dd>" in html
    assert "<dt>dict</dt>" in html
    assert "nested" in html


def test_export_units_to_html_preserves_whitespace_in_content():
    """Test that whitespace and line breaks in content are preserved."""
    content = """Line 1
Line 2
    Indented line
Line 4"""
    unit = _unit("unit-1", "Whitespace Test", content=content)
    html = export_units_to_html([unit])

    assert "<pre class=\"unit-content\">" in html
    assert "Line 1\nLine 2\n    Indented line\nLine 4</pre>" in html


def test_html_exporter_registered_in_registry():
    """Test that HTML exporter is registered in the export registry."""
    exporters = list_exporters()
    assert "html" in exporters


def test_get_html_exporter_from_registry():
    """Test retrieving HTML exporter from registry."""
    exporter = get_exporter("html")
    assert exporter is export_units_to_html


def test_html_exporter_works_through_registry():
    """Test that HTML export works when called through registry."""
    from graph.exports import export_units

    units = [_unit("unit-1", "Registry Test")]
    html = export_units("html", units, title="Registry Export")

    assert "<!DOCTYPE html>" in html
    assert "<h1>Registry Export</h1>" in html
    assert "Registry Test" in html
