"""HTML export adapter for knowledge unit collections."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from html import escape
from typing import Any

from graph.types.models import KnowledgeUnit

DEFAULT_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.6;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    color: #333;
}

.unit {
    margin-bottom: 40px;
    padding-bottom: 30px;
    border-bottom: 1px solid #e0e0e0;
}

.unit:last-child {
    border-bottom: none;
}

.unit-title {
    color: #1a1a1a;
    margin-bottom: 10px;
}

.unit-content {
    margin: 15px 0;
    white-space: pre-wrap;
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 4px;
    overflow-x: auto;
}

.unit-metadata {
    margin: 15px 0;
}

.unit-metadata dt {
    font-weight: 600;
    color: #555;
    margin-top: 8px;
}

.unit-metadata dd {
    margin-left: 20px;
    color: #666;
}

.unit-tags {
    margin: 15px 0;
}

.tag-badge {
    display: inline-block;
    background-color: #007bff;
    color: white;
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.9em;
    margin-right: 6px;
    margin-bottom: 6px;
}

.unit-timestamps {
    font-size: 0.9em;
    color: #888;
    margin-top: 10px;
}

.timestamp {
    margin-right: 15px;
}
"""


def export_units_to_html(
    units: Iterable[KnowledgeUnit],
    *,
    title: str = "Knowledge Units Export",
    css: str | None = None,
) -> str:
    """
    Export knowledge units to a formatted HTML5 document.

    Args:
        units: Iterable of KnowledgeUnit instances to export
        title: HTML document title (default: "Knowledge Units Export")
        css: Optional CSS styling to override default styles. If None, uses DEFAULT_CSS.

    Returns:
        Complete HTML5 document as a string
    """
    stylesheet = css if css is not None else DEFAULT_CSS
    units_list = list(units)

    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"    <title>{escape(title)}</title>",
        "    <style>",
        stylesheet,
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>{escape(title)}</h1>",
    ]

    for unit in units_list:
        html_parts.extend(_render_unit(unit))

    html_parts.extend([
        "</body>",
        "</html>",
    ])

    return "\n".join(html_parts)


def _render_unit(unit: KnowledgeUnit) -> list[str]:
    """Render a single knowledge unit as HTML."""
    parts = [
        '    <article class="unit">',
        f'        <h2 class="unit-title">{escape(unit.title or "Untitled")}</h2>',
    ]

    # Content
    if unit.content:
        parts.extend([
            f'        <pre class="unit-content">{escape(unit.content)}</pre>',
        ])

    # Metadata
    if unit.metadata:
        parts.append('        <dl class="unit-metadata">')
        for key, value in sorted(unit.metadata.items()):
            parts.extend([
                f"            <dt>{escape(str(key))}</dt>",
                f"            <dd>{escape(_format_value(value))}</dd>",
            ])
        parts.append("        </dl>")

    # Tags
    if unit.tags:
        parts.append('        <div class="unit-tags">')
        for tag in sorted(unit.tags):
            parts.append(f'            <span class="tag-badge">{escape(tag)}</span>')
        parts.append("        </div>")

    # Timestamps
    timestamp_parts = []
    if unit.created_at:
        timestamp_parts.append(
            f'<span class="timestamp">Created: {_format_datetime(unit.created_at)}</span>'
        )
    if unit.updated_at:
        timestamp_parts.append(
            f'<span class="timestamp">Updated: {_format_datetime(unit.updated_at)}</span>'
        )

    if timestamp_parts:
        parts.append(f'        <div class="unit-timestamps">{"".join(timestamp_parts)}</div>')

    parts.append("    </article>")
    return parts


def _format_value(value: Any) -> str:
    """Format a metadata value for display."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return str(value)
    return str(value)


def _format_datetime(dt: datetime | date | None) -> str:
    """Format a datetime for human-readable display."""
    if dt is None:
        return ""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    return dt.isoformat()
