"""Markdown export helpers for Kindle highlight review queues."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit


def export_units_to_kindle_review_queue_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    tag_filter: str | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic Kindle highlights review queue."""
    text = _render(list(units), tag_filter=tag_filter)
    if path is None:
        return text
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {"path": str(output_path), "bytes_written": output_path.stat().st_size}


def _render(units: list[KnowledgeUnit], *, tag_filter: str | None) -> str:
    groups: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        if tag_filter and tag_filter not in unit.tags:
            continue
        if not _is_kindle_highlight(unit):
            continue
        groups[(_text(unit.metadata.get("book_title")) or "Untitled", _text(unit.metadata.get("author")))].append(unit)

    lines = ["# Kindle Review Queue", ""]
    for (book_title, author), book_units in sorted(groups.items(), key=lambda item: (item[0][0].casefold(), item[0][1].casefold())):
        heading = f"{book_title} - {author}" if author else book_title
        lines.extend([f"## {heading}", ""])
        for unit in sorted(book_units, key=_highlight_sort_key):
            lines.append(f"- {unit.content}")
            note = _text(unit.metadata.get("note") or unit.metadata.get("note_text"))
            if note:
                lines.append(f"  Note: {note}")
            tags = ", ".join(sorted(unit.tags))
            if tags:
                lines.append(f"  Tags: {tags}")
            lines.append(f"  Source: {unit.source_id}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _is_kindle_highlight(unit: KnowledgeUnit) -> bool:
    project = _text(getattr(unit.source_project, "value", unit.source_project))
    clipping_type = _text(unit.metadata.get("clipping_type")).casefold()
    if clipping_type and clipping_type != "highlight":
        return False
    if project == "kindle":
        return clipping_type == "highlight" or unit.source_entity_type == "clipping"
    return bool(unit.metadata.get("book_title") and (clipping_type == "highlight" or unit.metadata.get("location")))


def _highlight_sort_key(unit: KnowledgeUnit) -> tuple[int, str, str]:
    return (_location_start(unit.metadata.get("location")), unit.created_at.isoformat(), unit.source_id)


def _location_start(value: Any) -> int:
    text = _text(value)
    digits = ""
    for char in text:
        if char.isdigit():
            digits += char
        elif digits:
            break
    return int(digits) if digits else 10**12


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())
