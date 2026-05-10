"""Logseq export adapter for knowledge unit collections."""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime

from graph.types.models import KnowledgeUnit


def export_units_to_logseq(
    units: Iterable[KnowledgeUnit],
    *,
    include_journal_pages: bool = True,
    include_block_ids: bool = True,
    include_queries: bool = False,
) -> str:
    """
    Export knowledge units to Logseq-compatible markdown format.

    Generates one markdown file per unit with Logseq-specific frontmatter,
    block references, and page links. For time-based units, also generates
    daily journal pages.

    Args:
        units: Iterable of KnowledgeUnit instances to export
        include_journal_pages: Whether to generate daily journal pages for time-based units
        include_block_ids: Whether to add block IDs for granular references
        include_queries: Whether to include embedded query blocks

    Returns:
        String containing all files with file path markers
        Format: "### FILE: path/to/file.md\n<content>\n\n"
    """
    files = export_units_to_logseq_files(
        units,
        include_journal_pages=include_journal_pages,
        include_block_ids=include_block_ids,
        include_queries=include_queries,
    )

    # Concatenate all files with clear markers
    parts: list[str] = []
    for file_path in sorted(files.keys()):
        parts.append(f"### FILE: {file_path}")
        parts.append(files[file_path])
        parts.append("")

    return "\n".join(parts)


def export_units_to_logseq_files(
    units: Iterable[KnowledgeUnit],
    *,
    include_journal_pages: bool = True,
    include_block_ids: bool = True,
    include_queries: bool = False,
) -> dict[str, str]:
    """
    Export knowledge units to Logseq-compatible markdown files.

    Returns a dictionary suitable for writing multiple files to disk.

    Args:
        units: Iterable of KnowledgeUnit instances to export
        include_journal_pages: Whether to generate daily journal pages for time-based units
        include_block_ids: Whether to add block IDs for granular references
        include_queries: Whether to include embedded query blocks

    Returns:
        Dictionary mapping file paths to markdown content
        Keys are relative paths like "pages/Unit Title.md" or "journals/2026_05_10.md"
    """
    units_list = list(units)
    files: dict[str, str] = {}

    # Group units by date for journal pages
    journal_units: dict[date, list[KnowledgeUnit]] = defaultdict(list)

    # Generate page file for each unit
    for unit in units_list:
        page_name = _sanitize_page_name(unit.title or unit.id)
        file_path = f"pages/{page_name}.md"
        files[file_path] = _render_unit_page(
            unit,
            include_block_ids=include_block_ids,
            include_queries=include_queries,
        )

        # Track for journal pages
        if include_journal_pages and unit.created_at:
            journal_date = unit.created_at.date()
            journal_units[journal_date].append(unit)

    # Generate journal pages
    if include_journal_pages:
        for journal_date, date_units in journal_units.items():
            journal_path = f"journals/{_format_journal_date(journal_date)}.md"
            files[journal_path] = _render_journal_page(
                journal_date,
                date_units,
                include_block_ids=include_block_ids,
            )

    return files


def _render_unit_page(
    unit: KnowledgeUnit,
    *,
    include_block_ids: bool = True,
    include_queries: bool = False,
) -> str:
    """Render a single unit as a Logseq page."""
    parts: list[str] = []

    # Frontmatter
    parts.append("---")
    parts.append(f"title: {unit.title or unit.id}")

    # Tags
    if unit.tags:
        formatted_tags = [_sanitize_tag(tag) for tag in unit.tags]
        parts.append(f"tags: {', '.join(formatted_tags)}")

    # Aliases (use source_id as alias)
    if unit.source_id != unit.id:
        parts.append(f"alias: {unit.source_id}")

    # Public flag (default to true for exports)
    parts.append("public: true")

    # Additional metadata as YAML
    if unit.metadata:
        for key, value in sorted(unit.metadata.items()):
            yaml_value = _format_metadata_value(value)
            parts.append(f"{key}: {yaml_value}")

    parts.append("---")
    parts.append("")

    # Content as blocks
    if unit.content:
        content_lines = unit.content.strip().split("\n")
        for line in content_lines:
            # Each line becomes a block
            block_text = f"- {line}"
            if include_block_ids:
                block_text += f"\n  id:: {_generate_block_id()}"
            parts.append(block_text)
        parts.append("")

    # Metadata as property blocks
    if unit.source_project:
        parts.append(f"- source-project:: [[{unit.source_project}]]")
        if include_block_ids:
            parts.append(f"  id:: {_generate_block_id()}")

    if unit.source_entity_type:
        parts.append(f"- source-type:: {unit.source_entity_type}")
        if include_block_ids:
            parts.append(f"  id:: {_generate_block_id()}")

    if unit.created_at:
        parts.append(f"- created:: [[{_format_logseq_date(unit.created_at)}]]")
        if include_block_ids:
            parts.append(f"  id:: {_generate_block_id()}")

    if unit.updated_at and unit.updated_at != unit.created_at:
        parts.append(f"- updated:: [[{_format_logseq_date(unit.updated_at)}]]")
        if include_block_ids:
            parts.append(f"  id:: {_generate_block_id()}")

    # Embedded query for related units (if enabled)
    if include_queries and unit.tags:
        parts.append("")
        parts.append("- ## Related")
        parts.append("  #+BEGIN_QUERY")
        # Query for units with the same tags
        tag_query = " OR ".join(f"(page-tags [[{_sanitize_tag(tag)}]])" for tag in unit.tags[:3])
        parts.append(f"  {{{{query {tag_query}}}}}")
        parts.append("  #+END_QUERY")
        if include_block_ids:
            parts.append(f"  id:: {_generate_block_id()}")

    return "\n".join(parts) + "\n"


def _render_journal_page(
    journal_date: date,
    units: list[KnowledgeUnit],
    *,
    include_block_ids: bool = True,
) -> str:
    """Render a daily journal page with links to units created on that date."""
    parts: list[str] = []

    # Frontmatter for journal page
    parts.append("---")
    parts.append(f"title: {_format_journal_title(journal_date)}")
    parts.append("public: true")
    parts.append("---")
    parts.append("")

    # Journal entry header
    parts.append(f"- ## {_format_journal_title(journal_date)}")
    if include_block_ids:
        parts.append(f"  id:: {_generate_block_id()}")

    # List units created on this date
    for unit in sorted(units, key=lambda u: u.created_at or datetime.min):
        page_link = _make_page_link(unit.title or unit.id)
        parts.append(f"- {page_link}")
        if include_block_ids:
            parts.append(f"  id:: {_generate_block_id()}")

        # Add tags as nested items
        if unit.tags:
            tag_links = " ".join(_make_page_link(tag) for tag in unit.tags)
            parts.append(f"  - tags: {tag_links}")
            if include_block_ids:
                parts.append(f"    id:: {_generate_block_id()}")

    return "\n".join(parts) + "\n"


def _sanitize_page_name(name: str) -> str:
    """
    Sanitize a page name for Logseq file names.

    Logseq uses the page title for file names, with special characters replaced.
    """
    # Replace slashes and colons which are problematic in filenames
    name = name.replace("/", "_")
    name = name.replace(":", "_")
    # Replace other characters that might cause issues
    name = re.sub(r'[<>"|?*]', "_", name)
    return name.strip()


def _sanitize_tag(tag: str) -> str:
    """
    Sanitize a tag for Logseq.

    Tags in Logseq should not contain special characters.
    """
    # Replace spaces and special chars with hyphens
    tag = re.sub(r"[\s/\\:]+", "-", tag)
    # Remove other problematic characters
    tag = re.sub(r'[<>"|?*#]', "", tag)
    return tag.strip("-")


def _make_page_link(title: str) -> str:
    """Create a Logseq page link with proper escaping."""
    # Logseq uses [[page name]] syntax
    # Escape closing brackets in the title
    escaped = title.replace("]]", r"\]\]")
    return f"[[{escaped}]]"


def _generate_block_id() -> str:
    """
    Generate a unique block ID for Logseq.

    Logseq uses short alphanumeric IDs for block references.
    Format: 8 hex characters (similar to UUID prefix)
    """
    return uuid.uuid4().hex[:8]


def _format_logseq_date(dt: datetime | date) -> str:
    """
    Format a date/datetime for Logseq page links.

    Logseq uses the format: "Mon dd, yyyy" (e.g., "May 10, 2026")
    """
    if isinstance(dt, datetime):
        dt = dt.date()
    return dt.strftime("%b %d, %Y")


def _format_journal_date(dt: date) -> str:
    """
    Format a date for journal file names.

    Logseq uses: YYYY_MM_DD format for journal files.
    """
    return dt.strftime("%Y_%m_%d")


def _format_journal_title(dt: date) -> str:
    """Format a date for journal page titles."""
    return _format_logseq_date(dt)


def _format_metadata_value(value: object) -> str:
    """Format a metadata value for YAML frontmatter."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, datetime):
        return f'"{value.isoformat()}"'
    if isinstance(value, date):
        return f'"{value.isoformat()}"'
    if isinstance(value, list):
        # YAML array format
        items = [_format_metadata_value(v) for v in value]
        return f"[{', '.join(items)}]"
    if isinstance(value, dict):
        # Simplified dict representation
        return str(value)
    if value is None:
        return "null"
    # String - quote if contains special characters
    str_value = str(value)
    if any(char in str_value for char in [":", "[", "]", "{", "}", "#", ","]):
        # Escape quotes
        str_value = str_value.replace('"', '\\"')
        return f'"{str_value}"'
    return str_value
