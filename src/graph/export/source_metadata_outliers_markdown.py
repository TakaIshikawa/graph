"""Markdown report for uncommon source metadata keys and values."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_source_metadata_outliers_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    max_key_frequency: float = 0.1,
    max_examples: int = 3,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report of uncommon metadata."""
    if not 0 < max_key_frequency <= 1:
        raise ValueError("max_key_frequency must be greater than 0 and at most 1")
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")

    unit_list = sorted(list(units), key=_unit_sort_key)
    groups = _groups(unit_list)
    report_groups = _reported_groups(groups, max_key_frequency=max_key_frequency, max_examples=max_examples)
    markdown = _render(report_groups, units_scanned=len(unit_list), max_key_frequency=max_key_frequency)

    if path is None:
        return markdown

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(unit_list),
        "groups_reported": len(report_groups),
        "bytes_written": output_path.stat().st_size,
    }


def _groups(units: list[KnowledgeUnit]) -> dict[tuple[str, str], list[KnowledgeUnit]]:
    grouped: dict[tuple[str, str], list[KnowledgeUnit]] = defaultdict(list)
    for unit in units:
        grouped[(_field_text(unit.source_project) or "Unknown", _inline_text(unit.source_entity_type) or "Unknown")].append(unit)
    return dict(sorted(grouped.items(), key=lambda item: (_sort_key(item[0][0]), _sort_key(item[0][1]))))


def _reported_groups(
    groups: dict[tuple[str, str], list[KnowledgeUnit]],
    *,
    max_key_frequency: float,
    max_examples: int,
) -> list[dict[str, Any]]:
    reported: list[dict[str, Any]] = []
    for (source_project, entity_type), group_units in groups.items():
        key_counts = Counter(
            _inline_text(key)
            for unit in group_units
            for key, value in (unit.metadata or {}).items()
            if _inline_text(key) and not _is_absent(value)
        )
        outlier_keys = [
            _key_report(key, count, group_units, max_examples)
            for key, count in sorted(key_counts.items(), key=lambda item: (_sort_key(item[0]), item[1]))
            if count / len(group_units) <= max_key_frequency
        ]
        if outlier_keys:
            reported.append(
                {
                    "source_project": source_project,
                    "entity_type": entity_type,
                    "unit_count": len(group_units),
                    "keys": outlier_keys,
                }
            )
    return reported


def _key_report(key: str, count: int, units: list[KnowledgeUnit], max_examples: int) -> dict[str, Any]:
    examples = [unit for unit in units if not _is_absent((unit.metadata or {}).get(key))]
    value_counts = Counter(_value_text((unit.metadata or {}).get(key)) for unit in examples)
    values = [
        {"value": value, "count": value_count}
        for value, value_count in sorted(value_counts.items(), key=lambda item: (_sort_key(item[0]), item[1]))
        if value
    ][:max_examples]
    return {
        "key": key,
        "count": count,
        "examples": [_unit_label(unit) for unit in sorted(examples, key=_unit_sort_key)[:max_examples]],
        "values": values,
    }


def _render(groups: list[dict[str, Any]], *, units_scanned: int, max_key_frequency: float) -> str:
    lines = [
        "# Source Metadata Outliers",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Groups reported | {len(groups)} |",
        f"| Max key frequency | {max_key_frequency:.2%} |",
    ]
    if not groups:
        lines.extend(["", "No metadata outliers found."])
        return "\n".join(lines).rstrip() + "\n"

    for group in groups:
        lines.extend(
            [
                "",
                f"## {_markdown_cell(group['source_project'])} / {_markdown_cell(group['entity_type'])}",
                "",
                f"Units: {group['unit_count']}",
                "",
                "| Key | Units | Example values | Example units |",
                "| --- | ---: | --- | --- |",
            ]
        )
        for key in group["keys"]:
            values = "; ".join(f"{item['value']} ({item['count']})" for item in key["values"]) or "_None_"
            examples = "; ".join(key["examples"]) or "_None_"
            lines.append(
                "| "
                f"{_markdown_cell(key['key'])} | "
                f"{key['count']} | "
                f"{_markdown_cell(values)} | "
                f"{_markdown_cell(examples)} |"
            )
    return "\n".join(lines).rstrip() + "\n"


def _is_absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not _inline_text(value)
    if isinstance(value, list | dict | tuple | set):
        return not value
    return False


def _value_text(value: Any) -> str:
    if isinstance(value, list | tuple | set):
        return ", ".join(_inline_text(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{_inline_text(key)}={_inline_text(val)}" for key, val in sorted(value.items()))
    return _inline_text(value)


def _unit_label(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.title) or _inline_text(unit.source_id) or _inline_text(unit.id) or "Untitled"


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (
        _field_text(unit.source_project),
        _inline_text(unit.source_entity_type),
        _inline_text(unit.title),
        _inline_text(unit.source_id),
    )


def _field_text(value: object) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: object) -> str:
    return _WHITESPACE_RE.sub(" ", "" if value is None else str(value)).strip()


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")
