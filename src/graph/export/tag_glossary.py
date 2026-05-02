"""Markdown tag glossary export helpers."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_tag_glossary_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path,
    *,
    min_count: int = 1,
    include_examples: int = 3,
) -> dict[str, Any]:
    """Write a deterministic Markdown glossary of tag usage."""
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    if (
        not isinstance(include_examples, int)
        or isinstance(include_examples, bool)
        or include_examples < 0
    ):
        raise ValueError("include_examples must be a non-negative integer")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_units = list(units)
    glossary = _build_glossary(all_units, min_count=min_count)
    report = _render_report(
        glossary,
        units_scanned=len(all_units),
        include_examples=include_examples,
    )
    output_path.write_text(report, encoding="utf-8")

    return {
        "path": str(output_path),
        "units_scanned": len(all_units),
        "tags_scanned": glossary["tags_scanned"],
        "tags_exported": len(glossary["tags"]),
        "min_count": min_count,
        "include_examples": include_examples,
        "bytes_written": output_path.stat().st_size,
    }


def _build_glossary(units: list[KnowledgeUnit], *, min_count: int) -> dict[str, Any]:
    tag_units: dict[str, list[KnowledgeUnit]] = defaultdict(list)
    for unit in sorted(units, key=_unit_sort_key):
        for tag in _unit_tags(unit):
            tag_units[tag].append(unit)

    exported_tags = {
        tag: tagged_units
        for tag, tagged_units in tag_units.items()
        if len(tagged_units) >= min_count
    }
    return {
        "tags_scanned": len(tag_units),
        "tags": {
            tag: {
                "units": sorted(tagged_units, key=_example_sort_key),
                "source_project_counts": Counter(
                    _field_value(unit.source_project) for unit in tagged_units
                ),
                "content_type_counts": Counter(
                    _field_value(unit.content_type) for unit in tagged_units
                ),
            }
            for tag, tagged_units in sorted(
                exported_tags.items(), key=lambda item: (item[0].casefold(), item[0])
            )
        },
    }


def _render_report(
    glossary: dict[str, Any],
    *,
    units_scanned: int,
    include_examples: int,
) -> str:
    lines = [
        "# Tag Glossary",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Tags scanned | {glossary['tags_scanned']} |",
        f"| Tags exported | {len(glossary['tags'])} |",
        "",
    ]

    if not glossary["tags"]:
        lines.append("_No tags matched the export criteria._")
        return "\n".join(lines).rstrip() + "\n"

    for tag, entry in glossary["tags"].items():
        lines.extend(
            [
                f"## {_heading_text(tag)}",
                "",
                f"- Usage count: {len(entry['units'])}",
                "",
                "### Source Projects",
                "",
                "| Source project | Count |",
                "| --- | ---: |",
                *_counter_rows(entry["source_project_counts"], empty_label="_None_"),
                "",
                "### Content Types",
                "",
                "| Content type | Count |",
                "| --- | ---: |",
                *_counter_rows(entry["content_type_counts"], empty_label="_None_"),
                "",
                "### Examples",
                "",
            ]
        )
        examples = entry["units"][:include_examples]
        if examples:
            for unit in examples:
                lines.append(
                    "- "
                    f"{_inline_markdown(_unit_label(unit))} "
                    f"(`{_code_text(unit.id)}`) "
                    f"- updated {_datetime_text(unit.updated_at)}"
                )
        else:
            lines.append("_No examples included._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _unit_tags(unit: KnowledgeUnit) -> list[str]:
    return sorted(
        {_inline_text(tag) for tag in unit.tags if _inline_text(tag)},
        key=lambda tag: (tag.casefold(), tag),
    )


def _counter_rows(counter: Counter[str], *, empty_label: str) -> list[str]:
    if not counter:
        return [f"| {empty_label} | 0 |"]
    return [
        f"| {_markdown_cell(key)} | {count} |"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _unit_label(unit: KnowledgeUnit) -> str:
    for value in (
        unit.metadata.get("label"),
        unit.title,
        unit.metadata.get("title"),
        unit.metadata.get("name"),
        unit.id,
        unit.source_id,
    ):
        text = _inline_text(value)
        if text:
            return text
    return "Untitled"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str]:
    return (
        _inline_text(unit.id),
        _field_value(unit.source_project),
        _inline_text(unit.source_id),
    )


def _example_sort_key(unit: KnowledgeUnit) -> tuple[str, str]:
    return (_reverse_datetime_text(unit.updated_at), _inline_text(unit.id))


def _reverse_datetime_text(value: object) -> str:
    text = _datetime_text(value)
    return "".join(chr(0x10FFFF - ord(char)) for char in text)


def _field_value(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _datetime_text(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return _inline_text(value)


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _heading_text(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("#", r"\#")


def _inline_markdown(value: object) -> str:
    return (
        _inline_text(value)
        .replace("\\", r"\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("(", r"\(")
        .replace(")", r"\)")
    )


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _code_text(value: object) -> str:
    return _inline_text(value).replace("`", r"\`")
