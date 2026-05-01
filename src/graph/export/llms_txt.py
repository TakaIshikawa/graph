"""llms.txt-style Markdown export helpers."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

from graph.types.models import KnowledgeUnit

_ANCHOR_RE = re.compile(r"[^a-zA-Z0-9_-]+")
_WHITESPACE_RE = re.compile(r"\s+")


def export_units_to_llms_txt(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
    *,
    title: str = "Graph Knowledge Base",
    max_units: int | None = None,
    include_metadata: bool = True,
) -> dict:
    """Write selected units as a compact llms.txt-style Markdown document."""
    if max_units is not None and max_units < 0:
        raise ValueError("max_units must be a non-negative integer")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = _ordered_units(units)
    exported_units = all_units[:max_units] if max_units is not None else all_units
    text = _render_llms_txt(
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


def _render_llms_txt(
    units: list[KnowledgeUnit],
    *,
    title: str,
    include_metadata: bool,
) -> str:
    lines = [
        f"# {_heading_text(title) or 'Graph Knowledge Base'}",
        "",
        "## Index",
        "",
    ]

    if units:
        for unit in units:
            lines.append(f"- [{_link_text(_unit_title(unit))}](#{_unit_anchor(unit)})")
    else:
        lines.append("_No units exported._")

    lines.extend(["", "## Units", ""])
    if not units:
        lines.extend(["_No units exported._", ""])
    for unit in units:
        lines.extend(_unit_section_lines(unit, include_metadata=include_metadata))

    return "\n".join(lines).rstrip() + "\n"


def _unit_section_lines(unit: KnowledgeUnit, *, include_metadata: bool) -> list[str]:
    lines = [
        f'<a id="{_unit_anchor(unit)}"></a>',
        "",
        f"### {_heading_text(_unit_title(unit))}",
        "",
    ]
    if include_metadata:
        lines.extend(
            [
                f"- ID: `{_code_text(unit.id)}`",
                (
                    f"- Source: {_inline_text(_field_value(unit.source_project))}/"
                    f"{_inline_text(unit.source_entity_type)}"
                    f" (`{_code_text(unit.source_id)}`)"
                ),
                f"- Type: `{_code_text(_field_value(unit.content_type))}`",
                f"- Tags: {_tags_text(unit.tags)}",
                "",
            ]
        )

    lines.extend([_fenced_text(_content_excerpt(unit.content)), ""])
    return lines


def _unit_title(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.title) or "Untitled"


def _unit_anchor(unit: KnowledgeUnit) -> str:
    text = _ANCHOR_RE.sub("-", _inline_text(unit.id)).strip("-").lower()
    return text or "unit"


def _content_excerpt(value: object, *, max_length: int = 1000) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value or "")).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3].rstrip() + "..."


def _field_value(value: object) -> str:
    return str(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _heading_text(value: object) -> str:
    text = _inline_text(value).replace("\\", "\\\\").replace("#", r"\#")
    return text


def _link_text(value: object) -> str:
    text = _inline_text(value)
    return (
        text.replace("\\", r"\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def _code_text(value: object) -> str:
    return _inline_text(value).replace("`", r"\`")


def _tags_text(tags: list[str]) -> str:
    if not tags:
        return "_None._"
    return ", ".join(f"`{_code_text(tag)}`" for tag in tags)


def _fenced_text(value: str) -> str:
    longest_run = max(
        (len(match.group(0)) for match in re.finditer(r"`+", value)),
        default=0,
    )
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}\n{value}\n{fence}"
