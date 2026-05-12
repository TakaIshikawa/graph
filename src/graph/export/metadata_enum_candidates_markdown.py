"""Markdown report for low-cardinality scalar metadata paths."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from graph.types.models import KnowledgeUnit

_WHITESPACE_RE = re.compile(r"\s+")


def export_metadata_enum_candidates_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    max_distinct_values: int = 12,
    min_units: int = 2,
    max_examples: int = 5,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report of enum-like metadata paths."""
    if (
        not isinstance(max_distinct_values, int)
        or isinstance(max_distinct_values, bool)
        or max_distinct_values < 1
    ):
        raise ValueError("max_distinct_values must be a positive integer")
    if not isinstance(min_units, int) or isinstance(min_units, bool) or min_units < 1:
        raise ValueError("min_units must be a positive integer")
    if not isinstance(max_examples, int) or isinstance(max_examples, bool) or max_examples < 0:
        raise ValueError("max_examples must be a non-negative integer")

    unit_list = sorted(list(units), key=_unit_sort_key)
    rows = _candidate_rows(
        unit_list,
        max_distinct_values=max_distinct_values,
        min_units=min_units,
        max_examples=max_examples,
    )
    text = _render_report(
        rows,
        units_scanned=len(unit_list),
        max_distinct_values=max_distinct_values,
        min_units=min_units,
        max_examples=max_examples,
    )

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(unit_list),
        "candidates_exported": len(rows),
        "max_distinct_values": max_distinct_values,
        "min_units": min_units,
        "max_examples": max_examples,
        "bytes_written": output_path.stat().st_size,
    }


def _candidate_rows(
    units: list[KnowledgeUnit],
    *,
    max_distinct_values: int,
    min_units: int,
    max_examples: int,
) -> list[dict[str, Any]]:
    path_units: dict[str, set[str]] = defaultdict(set)
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    value_counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    value_payloads: dict[tuple[str, str], Any] = {}

    for unit in units:
        if not isinstance(unit.metadata, Mapping):
            continue
        seen_paths: set[str] = set()
        for metadata_path, value in _flatten_metadata(unit.metadata):
            if not metadata_path or not _is_scalar(value):
                continue
            value_type = _value_type(value)
            normalized = _normalized_value(value)
            value_key = (
                value_type,
                json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str),
            )
            unit_key = _unit_id(unit)
            path_units[metadata_path].add(unit_key)
            if metadata_path not in seen_paths:
                source_counts[metadata_path][_unit_source(unit)] += 1
                seen_paths.add(metadata_path)
            type_counts[metadata_path][value_type] += 1
            value_counts[metadata_path][value_key] += 1
            value_payloads.setdefault(value_key, normalized)

    rows: list[dict[str, Any]] = []
    for metadata_path in path_units:
        unit_count = len(path_units[metadata_path])
        distinct_count = len(value_counts[metadata_path])
        if unit_count < min_units or distinct_count > max_distinct_values:
            continue
        examples = [
            _value_text(value_payloads[value_key])
            for value_key, _count in sorted(
                value_counts[metadata_path].items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )[:max_examples]
        ]
        rows.append(
            {
                "path": metadata_path,
                "unit_count": unit_count,
                "distinct_value_count": distinct_count,
                "value_types": _counter_text(type_counts[metadata_path]),
                "source_project_counts": _counter_text(source_counts[metadata_path]),
                "example_values": "; ".join(examples) if examples else "_None_",
            }
        )

    return sorted(rows, key=lambda row: (row["distinct_value_count"], -row["unit_count"], row["path"]))


def _flatten_metadata(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for raw_key, child in sorted(value.items(), key=lambda item: str(item[0])):
            key = str(raw_key).replace(".", "\\.")
            child_path = f"{prefix}.{key}" if prefix else key
            yield from _flatten_metadata(child, child_path)
        return
    if isinstance(value, list | tuple | set):
        return
    if prefix:
        yield prefix, value


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool | Enum)


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, Enum):
        return "string"
    return "string"


def _normalized_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _render_report(
    rows: list[dict[str, Any]],
    *,
    units_scanned: int,
    max_distinct_values: int,
    min_units: int,
    max_examples: int,
) -> str:
    lines = [
        "# Metadata Enum Candidates",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Candidates reported | {len(rows)} |",
        f"| Max distinct values | {max_distinct_values} |",
        f"| Min units | {min_units} |",
        f"| Max examples | {max_examples} |",
        "",
        "## Candidates",
        "",
        "| Metadata path | Units | Distinct values | Value types | Sources | Example values |",
        "| --- | ---: | ---: | --- | --- | --- |",
    ]
    if rows:
        for row in rows:
            lines.append(
                "| "
                f"{_markdown_cell(row['path'])} | "
                f"{row['unit_count']} | "
                f"{row['distinct_value_count']} | "
                f"{_markdown_cell(row['value_types'])} | "
                f"{_markdown_cell(row['source_project_counts'])} | "
                f"{_markdown_cell(row['example_values'])} |"
            )
    else:
        lines.append("| _None_ | 0 | 0 | _None_ | _None_ | _None_ |")
    return "\n".join(lines).rstrip() + "\n"


def _counter_text(counter: Counter[str]) -> str:
    if not counter:
        return "_None_"
    return "; ".join(
        f"{key} ({count})" for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


def _value_text(value: Any) -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        text = "null"
    elif isinstance(value, bool):
        text = str(value).lower()
    else:
        text = str(value)
    return _inline_text(text) or "_Blank_"


def _unit_source(unit: KnowledgeUnit) -> str:
    return _inline_text(getattr(unit.source_project, "value", unit.source_project)) or "Unknown"


def _unit_id(unit: KnowledgeUnit) -> str:
    return _inline_text(unit.id) or _inline_text(unit.source_id) or _inline_text(unit.title)


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (_unit_source(unit), _inline_text(unit.source_id), _inline_text(unit.title), _inline_text(unit.id))


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")
