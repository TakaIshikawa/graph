"""Org-mode export helpers for knowledge units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from pathlib import Path

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")
_ORG_SENSITIVE_RE = re.compile(
    r"(^\*+\s)|(^#\+)|(^:\s)|(^:PROPERTIES:\s*$)|(^:END:\s*$)|(\[\[[^\]]+\]\])",
    re.IGNORECASE | re.MULTILINE,
)
_BLOCK_END_RE = re.compile(r"^#\+END_", re.IGNORECASE)


def export_units_to_org(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
    *,
    title: str = "Knowledge Graph",
    include_metadata: bool = True,
    max_units: int | None = None,
) -> dict:
    """Write selected units as an Org-mode outline document."""
    if max_units is not None and max_units < 0:
        raise ValueError("max_units must be a non-negative integer")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = _ordered_units(units)
    exported_units = all_units[:max_units] if max_units is not None else all_units
    text = _render_org(
        exported_units,
        title=title,
        include_metadata=include_metadata,
    )
    output_path.write_text(text, encoding="utf-8")
    bytes_written = len(text.encode("utf-8"))

    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "units_exported": len(exported_units),
        "bytes_written": bytes_written,
    }


def _ordered_units(units: Iterable[KnowledgeUnit]) -> list[KnowledgeUnit]:
    if isinstance(units, Sequence):
        return list(units)
    return sorted(
        list(units),
        key=lambda unit: (
            _inline_text(unit.title or "Untitled").casefold(),
            _inline_text(unit.id),
        ),
    )


def _render_org(
    units: list[KnowledgeUnit],
    *,
    title: str,
    include_metadata: bool,
) -> str:
    lines = [f"#+TITLE: {_inline_text(title) or 'Knowledge Graph'}", ""]
    if not units:
        lines.append("# No units exported.")
    for unit in units:
        lines.extend(_unit_lines(unit, include_metadata=include_metadata))
    return "\n".join(lines).rstrip() + "\n"


def _unit_lines(unit: KnowledgeUnit, *, include_metadata: bool) -> list[str]:
    lines = [f"* {_heading_text(_unit_title(unit))}"]
    if include_metadata:
        lines.extend(
            [
                ":PROPERTIES:",
                f":source_project: {_property_value(_field_value(unit.source_project))}",
                f":source_id: {_property_value(unit.source_id)}",
                f":content_type: {_property_value(_field_value(unit.content_type))}",
                f":tags: {_property_value(_tags_text(unit.tags))}",
                f":created_at: {_property_value(_date_text(unit.created_at))}",
                f":updated_at: {_property_value(_date_text(unit.updated_at))}",
                ":END:",
            ]
        )

    content = str(unit.content or "").strip()
    if content:
        lines.extend(["", *_content_lines(content)])
    lines.append("")
    return lines


def _content_lines(content: str) -> list[str]:
    if _ORG_SENSITIVE_RE.search(content):
        return [
            "#+BEGIN_EXAMPLE",
            *[_example_line(line) for line in content.splitlines()],
            "#+END_EXAMPLE",
        ]
    return content.splitlines()


def _example_line(line: str) -> str:
    if _BLOCK_END_RE.match(line):
        return f",{line}"
    return line


def _unit_title(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.title) or "Untitled"


def _heading_text(value: object) -> str:
    text = _inline_text(value)
    return text.replace("\\", "\\\\").replace("\n", " ")


def _property_value(value: object) -> str:
    return _inline_text(value).replace("\n", " ")


def _tags_text(tags: list[str]) -> str:
    return ", ".join(sorted(_inline_text(tag) for tag in tags))


def _field_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _date_text(value: object) -> str:
    if isinstance(value, datetime | date):
        return value.isoformat()
    return _inline_text(value)


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()
