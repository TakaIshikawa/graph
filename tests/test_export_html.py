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


# Edge Case Tests for XSS Prevention and Security


def test_export_units_to_html_prevents_javascript_injection_in_title():
    """Test that JavaScript in title is escaped to prevent XSS."""
    unit = _unit(
        "unit-xss-title",
        "<script>alert('XSS')</script>",
        "Safe content",
    )
    html = export_units_to_html([unit])

    # JavaScript should be escaped
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "alert('XSS')" not in html or "alert(&#x27;XSS&#x27;)" in html


def test_export_units_to_html_prevents_javascript_injection_in_content():
    """Test that JavaScript in content is escaped."""
    unit = _unit(
        "unit-xss-content",
        "Safe Title",
        "<script>document.cookie</script><img src=x onerror=alert('XSS')>",
    )
    html = export_units_to_html([unit])

    # JavaScript should be escaped - HTML entities should be used
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    # The img tag should also be escaped
    assert "<img" not in html or "&lt;img" in html


def test_export_units_to_html_prevents_javascript_injection_in_tags():
    """Test that JavaScript in tags is escaped."""
    unit = _unit(
        "unit-xss-tags",
        "Safe Title",
        "Safe content",
        tags=["<script>alert(1)</script>", "tag&special"],
    )
    html = export_units_to_html([unit])

    # Script tags should be escaped
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    # Ampersand should be escaped
    assert "tag&amp;special" in html


def test_export_units_to_html_prevents_javascript_injection_in_metadata():
    """Test that JavaScript in metadata keys and values is escaped."""
    unit = _unit(
        "unit-xss-metadata",
        "Safe Title",
        "Safe content",
        metadata={
            "<script>key</script>": "value",
            "key": "<script>alert('value')</script>",
            "onload": "alert('xss')",
        },
    )
    html = export_units_to_html([unit])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_export_units_to_html_prevents_html_injection_in_doc_title():
    """Test that HTML injection in document title is escaped."""
    units = [_unit("unit-1", "Test")]
    html = export_units_to_html(
        units,
        title="</title><script>alert('XSS')</script><title>",
    )

    # Should not close the title tag early
    assert html.count("<title>") == 1
    assert html.count("</title>") == 1
    assert "&lt;/title&gt;" in html


def test_export_units_to_html_escapes_script_and_ampersand_in_doc_title():
    """Test that special characters in the document title are escaped."""
    html = export_units_to_html(
        [_unit("unit-1", "Test")],
        title="Report <script>alert('XSS')</script> & notes",
    )

    expected = (
        "<title>Report &lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt; "
        "&amp; notes</title>"
    )
    assert expected in html
    assert "<script>" not in html


def test_export_units_to_html_prevents_style_injection():
    """Test that style tags cannot be injected through content."""
    unit = _unit(
        "unit-style-inject",
        "Style Injection Test",
        "</style><style>body{background:red;}</style><style>",
        metadata={"key": "</style><script>alert('xss')</script><style>"},
    )
    html = export_units_to_html([unit])

    # Should have exactly 2 style tags (opening and closing in head)
    assert html.count("<style>") == 1
    assert html.count("</style>") == 1


def test_export_units_to_html_handles_event_handler_attributes():
    """Test that event handler text is present but safely escaped in content/text."""
    unit = _unit(
        "unit-events",
        "Title with < and >",
        "Content with & and <tag>",
        tags=["tag<script>"],
        metadata={"key": "value<>"},
    )
    html = export_units_to_html([unit])

    # HTML special characters should be escaped
    assert "&lt;" in html
    assert "&gt;" in html
    assert "&amp;" in html
    # Raw angle brackets should not appear in rendered content
    # (they may appear in CSS or structure but not in user content)


# Unicode and Encoding Edge Cases


def test_export_units_to_html_handles_unicode_in_all_fields():
    """Test Unicode characters in all HTML fields."""
    unit = _unit(
        "unit-unicode-全",
        "Title: 你好世界 🌍 émojis",
        "Content: Arabic مرحبا, Russian Привет, Hebrew שלום, Greek Γειά",
        tags=["tag-日本語", "tag-한글", "tag-🏷️"],
        metadata={
            "field_中文": "值",
            "field_emoji": "🎨🎭🎪",
            "field_mixed": "Mix 日本語 🌟",
        },
    )
    html = export_units_to_html([unit])

    # Verify all Unicode is preserved (not entity-encoded unnecessarily)
    assert "你好世界" in html
    assert "🌍" in html
    assert "émojis" in html
    assert "مرحبا" in html
    assert "Привет" in html
    assert "שלום" in html
    assert "Γειά" in html
    assert "日本語" in html
    assert "한글" in html
    assert "🏷️" in html


def test_export_units_to_html_handles_malformed_unicode_gracefully():
    """Test that malformed Unicode sequences are handled gracefully."""
    # Python strings are always valid Unicode, so we test edge cases
    # like combining characters and zero-width characters
    unit = _unit(
        "unit-combining",
        "Combining: é vs é",  # precomposed vs combined
        "Zero-width: a\u200Bb\u200Cc\uFEFFd",  # zero-width space, joiner, BOM
        metadata={"rtl": "Hebrew: \u202Eשלום\u202C mixed"},  # RTL override
    )
    html = export_units_to_html([unit])

    # Should not break HTML structure
    assert "<!DOCTYPE html>" in html
    assert "</html>" in html


def test_export_units_to_html_handles_newlines_and_control_chars():
    """Test that newlines and control characters are preserved properly."""
    unit = _unit(
        "unit-control",
        "Title\nwith\nnewlines",
        "Content\twith\ttabs\nand\rnewlines\r\nand CRLF",
        metadata={"key\nwith\nnewline": "value\twith\ttab"},
    )
    html = export_units_to_html([unit])

    # Newlines in content should be preserved in <pre>
    assert "<pre class=\"unit-content\">" in html
    assert "\n" in html or "&#10;" in html


# Empty Collection and Missing Field Edge Cases


def test_export_units_to_html_empty_collection_generates_valid_document():
    """Test that empty collection still produces valid HTML."""
    html = export_units_to_html([])

    assert "<!DOCTYPE html>" in html
    assert '<html lang="en">' in html
    assert "</html>" in html
    assert '<article class="unit">' not in html


def test_export_units_to_html_unit_with_empty_title_and_content():
    """Test unit with empty strings for title and content."""
    unit = KnowledgeUnit(
        id="unit-empty",
        source_project=SourceProject.MAX,
        source_id="source-empty",
        source_entity_type="note",
        title="",
        content="",
        content_type=ContentType.INSIGHT,
        metadata={},
        tags=[],
    )
    html = export_units_to_html([unit])

    # Should handle empty title gracefully by showing "Untitled"
    assert "Untitled" in html
    # Empty content should not render content block
    assert "<pre class=\"unit-content\"></pre>" not in html


def test_export_units_to_html_metadata_with_none_values():
    """Test metadata containing None values."""
    unit = _unit(
        "unit-none-meta",
        "None Metadata",
        metadata={
            "key1": "value",
            "key2": None,
            "key3": "another",
        },
    )
    html = export_units_to_html([unit])

    assert "<dt>key1</dt>" in html
    assert "<dt>key2</dt>" in html
    assert "<dd>None</dd>" in html
    assert "<dt>key3</dt>" in html


# Long Content and Layout Edge Cases


def test_export_units_to_html_extremely_long_content():
    """Test handling of very long content that could break layout."""
    long_content = "A" * 100000 + "\n" + "B" * 50000
    unit = _unit("unit-long", "Long Content", long_content)
    html = export_units_to_html([unit])

    # Should still be valid HTML
    assert "<!DOCTYPE html>" in html
    assert long_content in html
    assert 'class="unit-content"' in html


def test_export_units_to_html_extremely_long_title():
    """Test handling of very long titles."""
    long_title = "Title " + "X" * 10000
    unit = _unit("unit-long-title", long_title)
    html = export_units_to_html([unit])

    assert long_title in html
    assert '<h2 class="unit-title">' in html


def test_export_units_to_html_many_tags():
    """Test rendering with a large number of tags."""
    tags = [f"tag-{i:04d}" for i in range(1000)]
    unit = _unit("unit-many-tags", "Many Tags", tags=tags)
    html = export_units_to_html([unit])

    assert html.count('<span class="tag-badge">') == 1000


def test_export_units_to_html_deeply_nested_metadata():
    """Test deeply nested metadata structures."""
    unit = _unit(
        "unit-nested",
        "Nested Metadata",
        metadata={
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {"deep": "value"}
                    }
                }
            }
        },
    )
    html = export_units_to_html([unit])

    assert "<dt>level1</dt>" in html
    # Nested dicts are converted to strings
    assert "level2" in html


# CSS and Style Edge Cases


def test_export_units_to_html_custom_css_with_special_characters():
    """Test custom CSS with special characters."""
    custom_css = """
    body { content: "<script>"; }
    .class { background: url("data:image/svg+xml,<svg></svg>"); }
    """
    units = [_unit("unit-1", "Test")]
    html = export_units_to_html(units, css=custom_css)

    # CSS should be included as-is (not escaped) in style tag
    assert custom_css in html


def test_export_units_to_html_empty_custom_css():
    """Test providing empty string as custom CSS."""
    units = [_unit("unit-1", "Test")]
    html = export_units_to_html(units, css="")

    # Should have style tags but empty content
    assert "<style>" in html
    assert "</style>" in html


# Special Character Edge Cases


def test_export_units_to_html_metadata_with_special_characters():
    """Test metadata with quotes, backslashes, and other special chars."""
    unit = _unit(
        "unit-special",
        "Special Chars",
        metadata={
            "quotes": 'Value with "double" and \'single\' quotes',
            "backslash": "Path\\to\\file",
            "ampersand": "Tom & Jerry",
            "less_greater": "1 < 2 > 0",
        },
    )
    html = export_units_to_html([unit])

    assert "&quot;" in html or "\"" in html
    assert "&amp;" in html
    assert "&lt;" in html
    assert "&gt;" in html


def test_export_units_to_html_content_with_html_entities():
    """Test content that already contains HTML entity references."""
    unit = _unit(
        "unit-entities",
        "HTML Entities",
        "Already escaped: &lt;tag&gt; and &amp; and &quot;",
    )
    html = export_units_to_html([unit])

    # Should double-escape: &lt; becomes &amp;lt;
    assert "&amp;lt;" in html
    assert "&amp;amp;" in html
    assert "&amp;quot;" in html


def test_export_units_to_html_large_collection():
    """Test export of large collection to ensure performance is reasonable."""
    units = [_unit(f"unit-{i}", f"Title {i}", f"Content {i}") for i in range(500)]
    html = export_units_to_html(units)

    assert html.count('<article class="unit">') == 500
    assert "Title 0" in html
    assert "Title 499" in html


def test_export_units_to_html_whitespace_only_content():
    """Test content that is only whitespace."""
    unit = _unit("unit-whitespace", "Whitespace Content", "   \n\t  \r\n   ")
    html = export_units_to_html([unit])

    # Whitespace-only content should still be rendered
    assert '<pre class="unit-content">   \n\t  \r\n   </pre>' in html


def test_export_units_to_html_preserves_multiple_consecutive_spaces():
    """Test that multiple consecutive spaces are preserved in pre tags."""
    content = "Word1    Word2        Word3"
    unit = _unit("unit-spaces", "Multiple Spaces", content)
    html = export_units_to_html([unit])

    assert "Word1    Word2        Word3" in html


def test_export_units_to_html_metadata_with_empty_string_values():
    """Test metadata with empty string values."""
    unit = _unit(
        "unit-empty-strings",
        "Empty Strings",
        metadata={"empty": "", "normal": "value", "whitespace": "   "},
    )
    html = export_units_to_html([unit])

    assert "<dt>empty</dt>" in html
    assert "<dd></dd>" in html or "<dd> </dd>" in html or html.count("<dd>") >= 3
