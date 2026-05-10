"""Roam Research export adapter for knowledge unit collections."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import date, datetime

from graph.types.models import KnowledgeUnit


def export_units_to_roam(
    units: Iterable[KnowledgeUnit],
    *,
    format: str = "json",
    include_attributes: bool = True,
    include_block_refs: bool = True,
) -> str:
    """
    Export knowledge units to Roam Research-compatible format.

    Generates Roam pages with hierarchical bullet points, block references,
    and attributes. Supports both JSON and EDN output formats.

    Args:
        units: Iterable of KnowledgeUnit instances to export
        format: Output format - "json" or "edn"
        include_attributes: Whether to include metadata as attributes (:key:: value)
        include_block_refs: Whether to generate block UIDs for references

    Returns:
        String in the specified format (JSON or EDN)
    """
    if format not in ("json", "edn"):
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'edn'.")

    pages = export_units_to_roam_pages(
        units,
        include_attributes=include_attributes,
        include_block_refs=include_block_refs,
    )

    if format == "json":
        return json.dumps(pages, indent=2, ensure_ascii=False)
    else:
        return _serialize_to_edn(pages)


def export_units_to_roam_pages(
    units: Iterable[KnowledgeUnit],
    *,
    include_attributes: bool = True,
    include_block_refs: bool = True,
) -> list[dict]:
    """
    Export knowledge units to Roam page structures.

    Returns a list of page dictionaries suitable for Roam import.

    Args:
        units: Iterable of KnowledgeUnit instances to export
        include_attributes: Whether to include metadata as attributes
        include_block_refs: Whether to generate block UIDs

    Returns:
        List of page dictionaries with Roam structure
    """
    units_list = list(units)
    pages: list[dict] = []

    # Create one page per unit
    for unit in units_list:
        page = _create_roam_page(
            unit,
            include_attributes=include_attributes,
            include_block_refs=include_block_refs,
        )
        pages.append(page)

    return pages


def _create_roam_page(
    unit: KnowledgeUnit,
    *,
    include_attributes: bool = True,
    include_block_refs: bool = True,
) -> dict:
    """Create a Roam page structure from a knowledge unit."""
    page_title = unit.title or unit.id
    children: list[dict] = []

    # Add content as nested bullets
    if unit.content:
        content_blocks = _create_content_blocks(
            unit.content,
            include_block_refs=include_block_refs,
        )
        children.extend(content_blocks)

    # Add metadata as attributes
    if include_attributes:
        # Source project attribute
        if unit.source_project:
            children.append(
                _create_attribute_block(
                    "source-project",
                    f"[[{unit.source_project}]]",
                    include_block_refs=include_block_refs,
                )
            )

        # Source type attribute
        if unit.source_entity_type:
            children.append(
                _create_attribute_block(
                    "source-type",
                    unit.source_entity_type,
                    include_block_refs=include_block_refs,
                )
            )

        # Created date attribute
        if unit.created_at:
            children.append(
                _create_attribute_block(
                    "created",
                    _format_roam_date(unit.created_at),
                    include_block_refs=include_block_refs,
                )
            )

        # Updated date attribute
        if unit.updated_at and unit.updated_at != unit.created_at:
            children.append(
                _create_attribute_block(
                    "updated",
                    _format_roam_date(unit.updated_at),
                    include_block_refs=include_block_refs,
                )
            )

        # Custom metadata as attributes
        if unit.metadata:
            for key, value in sorted(unit.metadata.items()):
                formatted_value = _format_attribute_value(value)
                children.append(
                    _create_attribute_block(
                        key,
                        formatted_value,
                        include_block_refs=include_block_refs,
                    )
                )

    # Tags as page references
    if unit.tags:
        tag_refs = " ".join(_make_page_reference(tag) for tag in unit.tags)
        children.append(
            _create_block(
                f"tags:: {tag_refs}",
                include_block_refs=include_block_refs,
            )
        )

    # Roam page structure
    page: dict = {
        "title": page_title,
        "children": children,
    }

    # Add edit metadata
    if unit.created_at:
        page["create-time"] = int(unit.created_at.timestamp() * 1000)
    if unit.updated_at:
        page["edit-time"] = int(unit.updated_at.timestamp() * 1000)

    return page


def _create_content_blocks(
    content: str,
    *,
    include_block_refs: bool = True,
) -> list[dict]:
    """Create nested block structure from content text."""
    lines = content.strip().split("\n")
    blocks: list[dict] = []

    for line in lines:
        if line.strip():
            blocks.append(_create_block(line, include_block_refs=include_block_refs))

    return blocks


def _create_block(
    text: str,
    *,
    children: list[dict] | None = None,
    include_block_refs: bool = True,
) -> dict:
    """Create a Roam block with optional children."""
    block: dict = {
        "string": text,
    }

    if include_block_refs:
        block["uid"] = _generate_block_uid()

    if children:
        block["children"] = children

    return block


def _create_attribute_block(
    key: str,
    value: str,
    *,
    include_block_refs: bool = True,
) -> dict:
    """Create a Roam attribute block with :key:: value syntax."""
    # Roam attribute syntax
    text = f"{key}:: {value}"
    return _create_block(text, include_block_refs=include_block_refs)


def _generate_block_uid() -> str:
    """
    Generate a unique block UID for Roam.

    Roam uses 9-character alphanumeric UIDs.
    Format: base-36 encoded timestamp + random suffix
    """
    # Use first 9 characters of a UUID (alphanumeric)
    uid = uuid.uuid4().hex[:9]
    return uid


def _make_page_reference(title: str) -> str:
    """Create a Roam page reference with proper escaping."""
    # Roam uses [[page name]] syntax
    # Escape closing brackets in the title
    escaped = title.replace("]]", r"\]\]")
    return f"[[{escaped}]]"


def _format_roam_date(dt: datetime | date) -> str:
    """
    Format a date/datetime for Roam page references.

    Roam uses the format: "Month DDth, YYYY" (e.g., "May 10th, 2026")
    """
    if isinstance(dt, datetime):
        dt = dt.date()

    # Get day with ordinal suffix
    day = dt.day
    if 4 <= day <= 20 or 24 <= day <= 30:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

    # Format: "May 10th, 2026"
    return dt.strftime(f"%B {day}{suffix}, %Y")


def _format_attribute_value(value: object) -> str:
    """Format a metadata value for Roam attributes."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime):
        return _format_roam_date(value)
    if isinstance(value, date):
        return _format_roam_date(value)
    if isinstance(value, list):
        # Format as comma-separated list
        items = [_format_attribute_value(v) for v in value]
        return ", ".join(items)
    if isinstance(value, dict):
        # Simplified dict representation
        return str(value)
    if value is None:
        return ""
    return str(value)


def _serialize_to_edn(pages: list[dict]) -> str:
    """
    Serialize pages to EDN (Extensible Data Notation) format.

    EDN is Roam's native format, similar to Clojure data structures.
    """
    lines: list[str] = []
    lines.append("[")

    for i, page in enumerate(pages):
        lines.append(_page_to_edn(page))
        if i < len(pages) - 1:
            lines.append("")

    lines.append("]")
    return "\n".join(lines)


def _page_to_edn(page: dict) -> str:
    """Convert a single page to EDN format."""
    parts: list[str] = ["{"]

    # Title
    parts.append(f' :title "{_escape_edn_string(page["title"])}"')

    # Timestamps
    if "create-time" in page:
        parts.append(f' :create-time {page["create-time"]}')
    if "edit-time" in page:
        parts.append(f' :edit-time {page["edit-time"]}')

    # Children blocks
    if page.get("children"):
        parts.append(" :children [")
        for child in page["children"]:
            parts.append(_block_to_edn(child, indent=2))
        parts.append(" ]")

    parts.append("}")
    return "\n".join(parts)


def _block_to_edn(block: dict, indent: int = 0) -> str:
    """Convert a block to EDN format."""
    prefix = " " * indent
    parts: list[str] = [f"{prefix}{{"]

    # String content
    parts.append(f'{prefix} :string "{_escape_edn_string(block["string"])}"')

    # UID
    if "uid" in block:
        parts.append(f'{prefix} :uid "{block["uid"]}"')

    # Children (recursive)
    if block.get("children"):
        parts.append(f"{prefix} :children [")
        for child in block["children"]:
            parts.append(_block_to_edn(child, indent=indent + 2))
        parts.append(f"{prefix} ]")

    parts.append(f"{prefix}}}")
    return "\n".join(parts)


def _escape_edn_string(text: str) -> str:
    """Escape special characters in EDN strings."""
    # Escape quotes and backslashes
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    # Escape newlines
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    return text
