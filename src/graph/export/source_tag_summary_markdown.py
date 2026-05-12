"""Markdown export for tag usage grouped by source."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_source_tag_summary_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    min_count: int = 1,
    limit_per_source: int | None = None,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown summary of tag counts by source."""
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if limit_per_source is not None and (
        not isinstance(limit_per_source, int)
        or isinstance(limit_per_source, bool)
        or limit_per_source < 1
    ):
        raise ValueError("limit_per_source must be a positive integer or None")

    unit_list = list(units)
    sections = _source_sections(unit_list, min_count=min_count, limit_per_source=limit_per_source)
    text = _render_report(sections, min_count=min_count, limit_per_source=limit_per_source)

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "sources_exported": len(sections),
        "units_scanned": len(unit_list),
        "tag_rows_exported": sum(len(section["rows"]) for section in sections),
        "min_count": min_count,
        "limit_per_source": limit_per_source,
        "bytes_written": output_path.stat().st_size,
    }


def _source_sections(
    units: list[KnowledgeUnit],
    *,
    min_count: int,
    limit_per_source: int | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        grouped[_unit_source(unit)].append(unit)

    sections: list[dict[str, Any]] = []
    for source in sorted(grouped, key=_sort_key):
        source_units = sorted(grouped[source], key=_unit_sort_key)
        counts, labels = _tag_counts(source_units)
        rows = [
            {"tag": labels[key], "count": count}
            for key, count in sorted(
                counts.items(),
                key=lambda item: (-item[1], labels[item[0]].casefold(), labels[item[0]]),
            )
            if count >= min_count
        ]
        if limit_per_source is not None:
            rows = rows[:limit_per_source]
        sections.append(
            {
                "source": source,
                "total_units": len(source_units),
                "tagged_units": sum(1 for unit in source_units if _unit_tags(unit)),
                "rows": rows,
            }
        )
    return sections


def _render_report(
    sections: list[dict[str, Any]],
    *,
    min_count: int,
    limit_per_source: int | None,
) -> str:
    lines = ["# Source Tag Summary", ""]
    lines.append(f"- Min count: {min_count}")
    if limit_per_source is not None:
        lines.append(f"- Limit per source: {limit_per_source}")
    lines.append("")

    if not sections:
        lines.extend(["_No units exported._", ""])
        return "\n".join(lines)

    for section in sections:
        lines.extend(
            [
                f"## {_markdown_text(section['source'])}",
                "",
                f"- Total units: {section['total_units']}",
                f"- Tagged units: {section['tagged_units']}",
                "",
                "| Tag | Count |",
                "| --- | ---: |",
            ]
        )
        rows = section["rows"]
        if rows:
            for row in rows:
                lines.append(f"| {_table_text(row['tag'])} | {row['count']} |")
        else:
            lines.append("| _None_ | 0 |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _tag_counts(units: Iterable[KnowledgeUnit]) -> tuple[Counter[str], dict[str, str]]:
    counts: Counter[str] = Counter()
    labels: dict[str, str] = {}
    for unit in units:
        for tag in _unit_tags(unit):
            key = tag.casefold()
            counts[key] += 1
            current = labels.get(key)
            if current is None or _sort_key(tag) < _sort_key(current):
                labels[key] = tag
    return counts, labels


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted({_inline_text(tag) for tag in unit.tags if _inline_text(tag)}, key=_sort_key)


def _unit_source(unit: KnowledgeUnit) -> str:
    return _field_value(unit.source_project) or "Unknown"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (_unit_source(unit).casefold(), _inline_text(unit.source_id), _inline_text(unit.id))


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _markdown_text(value: object) -> str:
    return _inline_text(value).replace("\\", r"\\").replace("*", r"\*").replace("_", r"\_")


def _table_text(value: object) -> str:
    return _markdown_text(value).replace("|", r"\|")
