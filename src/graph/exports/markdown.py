"""Markdown export adapter for knowledge unit collections."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime

from graph.types.models import KnowledgeUnit


def export_units_to_markdown(
    units: Iterable[KnowledgeUnit],
    *,
    title: str = "Knowledge Graph Export",
    include_toc: bool = True,
    include_metadata: bool = True,
    include_edges: bool = False,
    edges: list[tuple[str, str, str]] | None = None,
) -> str:
    """
    Export knowledge units as structured Markdown documentation.

    Units are grouped by source_project, then by tags within each project group.

    Args:
        units: Iterable of KnowledgeUnit instances to export
        title: Document title
        include_toc: Whether to generate a table of contents
        include_metadata: Whether to include metadata tables for units
        include_edges: Whether to render edge relationships (requires edges param)
        edges: List of (from_id, to_id, relation) tuples for relationship rendering

    Returns:
        Markdown formatted string
    """
    units_list = list(units)
    edges = edges or []

    parts: list[str] = []

    # Title
    parts.append(f"# {_escape_md(title)}")
    parts.append("")

    if not units_list:
        parts.append("*No knowledge units to export.*")
        parts.append("")
        return "\n".join(parts)

    # Group units by source_project
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units_list:
        project = str(unit.source_project)
        grouped[project].append(unit)

    # Table of contents
    if include_toc:
        parts.append("## Table of Contents")
        parts.append("")
        for project in sorted(grouped.keys()):
            anchor = _make_anchor(project)
            parts.append(f"- [{_escape_md(project)}](#{anchor})")
            # Sub-entries for tags within this project
            project_tags = _collect_tags(grouped[project])
            for tag in project_tags:
                tag_anchor = _make_anchor(f"{project}-{tag}")
                parts.append(f"  - [{_escape_md(tag)}](#{tag_anchor})")
        parts.append("")

    # Sections per project
    for project in sorted(grouped.keys()):
        project_units = grouped[project]
        parts.append(f"## {_escape_md(project)}")
        parts.append("")

        # Group by tags within project
        tagged: dict[str, list[KnowledgeUnit]] = defaultdict(list)
        untagged: list[KnowledgeUnit] = []
        for unit in project_units:
            if unit.tags:
                for tag in sorted(unit.tags):
                    tagged[tag].append(unit)
            else:
                untagged.append(unit)

        # Render tagged groups
        for tag in sorted(tagged.keys()):
            tag_units = tagged[tag]
            parts.append(f"### {_escape_md(tag)}")
            parts.append("")
            for unit in tag_units:
                parts.extend(_render_unit(unit, include_metadata=include_metadata))
            parts.append("")

        # Render untagged units
        if untagged:
            parts.append("### Untagged")
            parts.append("")
            for unit in untagged:
                parts.extend(_render_unit(unit, include_metadata=include_metadata))
            parts.append("")

    # Edge relationships
    if include_edges and edges:
        parts.append("## Relationships")
        parts.append("")
        parts.append("```mermaid")
        parts.append("graph LR")
        # Build id->title lookup
        title_map = {u.id: u.title or u.id for u in units_list}
        for from_id, to_id, relation in edges:
            from_label = _escape_mermaid(title_map.get(from_id, from_id))
            to_label = _escape_mermaid(title_map.get(to_id, to_id))
            rel_label = _escape_mermaid(relation)
            parts.append(f"    {_mermaid_id(from_id)}[{from_label}] -->|{rel_label}| {_mermaid_id(to_id)}[{to_label}]")
        parts.append("```")
        parts.append("")

    return "\n".join(parts)


def _render_unit(unit: KnowledgeUnit, *, include_metadata: bool = True) -> list[str]:
    """Render a single unit as Markdown."""
    parts: list[str] = []

    # Unit heading (h4)
    unit_title = unit.title or "Untitled"
    parts.append(f"#### {_escape_md(unit_title)}")
    parts.append("")

    # Content
    if unit.content:
        parts.append(unit.content)
        parts.append("")

    # Metadata table
    if include_metadata and unit.metadata:
        parts.append("| Field | Value |")
        parts.append("|-------|-------|")
        for key in sorted(unit.metadata.keys()):
            value = unit.metadata[key]
            parts.append(f"| {_escape_md_table(str(key))} | {_escape_md_table(_format_value(value))} |")
        parts.append("")

    # Tags as inline badges
    if unit.tags:
        tag_str = " ".join(f"`{tag}`" for tag in sorted(unit.tags))
        parts.append(f"**Tags:** {tag_str}")
        parts.append("")

    # Timestamps
    timestamps: list[str] = []
    if unit.created_at:
        timestamps.append(f"Created: {_format_datetime(unit.created_at)}")
    if unit.updated_at:
        timestamps.append(f"Updated: {_format_datetime(unit.updated_at)}")
    if timestamps:
        parts.append(f"*{' | '.join(timestamps)}*")
        parts.append("")

    parts.append("---")
    parts.append("")
    return parts


def _collect_tags(units: list[KnowledgeUnit]) -> list[str]:
    """Collect and sort unique tags from a list of units."""
    tags: set[str] = set()
    for unit in units:
        tags.update(unit.tags)
    return sorted(tags)


def _escape_md(text: str) -> str:
    """Escape Markdown special characters in inline text."""
    # Escape characters that have meaning in Markdown
    special = r"\\`*_{}[]()#+-.!|~>"
    result = []
    for char in text:
        if char in special:
            result.append(f"\\{char}")
        else:
            result.append(char)
    return "".join(result)


def _escape_md_table(text: str) -> str:
    """Escape text for use inside Markdown table cells."""
    # Pipes must be escaped in tables, and newlines replaced
    text = text.replace("|", "\\|")
    text = text.replace("\n", " ")
    text = text.replace("\r", "")
    return text


def _escape_mermaid(text: str) -> str:
    """Escape text for Mermaid diagram labels."""
    # Mermaid uses quotes for special chars
    text = text.replace('"', "'")
    text = text.replace("[", "(")
    text = text.replace("]", ")")
    return text


def _mermaid_id(unit_id: str) -> str:
    """Convert a unit ID to a valid Mermaid node ID."""
    # Replace non-alphanumeric chars with underscores
    return re.sub(r"[^a-zA-Z0-9]", "_", unit_id)


def _make_anchor(text: str) -> str:
    """Create a GitHub-flavored Markdown heading anchor."""
    # Lowercase, replace spaces with hyphens, remove non-alphanumeric (except hyphens)
    anchor = text.lower().strip()
    anchor = re.sub(r"\s+", "-", anchor)
    anchor = re.sub(r"[^a-z0-9\-_]", "", anchor)
    return anchor


def _format_value(value: object) -> str:
    """Format a metadata value for display."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return str(value)
    if value is None:
        return ""
    return str(value)


def _format_datetime(dt: datetime | date | None) -> str:
    """Format a datetime for human-readable display."""
    if dt is None:
        return ""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    return dt.isoformat()
