"""Markdown schema inventory reports for knowledge unit metadata."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from graph.types.models import KnowledgeUnit

_MAX_EXAMPLES_PER_KEY = 3
_MAX_EXAMPLE_LENGTH = 80
_WHITESPACE_RE = re.compile(r"\s+")


def export_unit_schema_inventory(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
) -> str:
    """Return a deterministic Markdown report of unit metadata schema usage."""
    all_units = sorted(list(units), key=_unit_sort_key)
    inventory = _build_inventory(all_units)
    report = _render_report(inventory)

    if path is not None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    return report


def _build_inventory(units: list[KnowledgeUnit]) -> dict[str, Any]:
    key_counts: Counter[str] = Counter()
    key_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    key_examples: dict[str, set[str]] = defaultdict(set)
    type_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    source_project_counts: Counter[str] = Counter()
    content_type_counts: Counter[str] = Counter()

    for unit in units:
        source_project_counts[_field_value(unit.source_project)] += 1
        content_type_counts[_field_value(unit.content_type)] += 1
        tag_counts.update(_inline_text(tag) for tag in unit.tags if _inline_text(tag))

        for key, value in _flatten_metadata(unit.metadata):
            value_type = _value_type(value)
            key_counts[key] += 1
            key_type_counts[key][value_type] += 1
            type_counts[value_type] += 1
            if len(key_examples[key]) < _MAX_EXAMPLES_PER_KEY:
                key_examples[key].add(_example_value(value))

    return {
        "unit_count": len(units),
        "key_counts": key_counts,
        "key_type_counts": key_type_counts,
        "key_examples": key_examples,
        "type_counts": type_counts,
        "tag_counts": tag_counts,
        "source_project_counts": source_project_counts,
        "content_type_counts": content_type_counts,
    }


def _flatten_metadata(
    metadata: Mapping[Any, Any],
    prefix: str = "",
) -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = []
    for raw_key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        key = _path_part(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            items.extend(_flatten_metadata(value, path))
        else:
            items.append((path, value))
    return items


def _render_report(inventory: dict[str, Any]) -> str:
    lines = [
        "# Unit Schema Inventory",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Units scanned | {inventory['unit_count']} |",
        f"| Metadata keys | {len(inventory['key_counts'])} |",
        "",
        "## Metadata Keys",
        "",
        "| Metadata key | Count | Types | Examples |",
        "| --- | ---: | --- | --- |",
    ]

    if inventory["key_counts"]:
        for key in sorted(inventory["key_counts"]):
            type_text = _counter_text(inventory["key_type_counts"][key])
            examples = sorted(inventory["key_examples"][key])[:_MAX_EXAMPLES_PER_KEY]
            lines.append(
                "| "
                f"{_markdown_cell(key)} | "
                f"{inventory['key_counts'][key]} | "
                f"{_markdown_cell(type_text)} | "
                f"{_markdown_cell('; '.join(examples))} |"
            )
    else:
        lines.append("| _None_ | 0 | _None_ | _None_ |")

    lines.extend(
        [
            "",
            "## Value Types",
            "",
            "| Type | Count |",
            "| --- | ---: |",
            *_counter_rows(inventory["type_counts"], empty_label="_None_"),
            "",
            "## Top Tags",
            "",
            "| Tag | Count |",
            "| --- | ---: |",
            *_counter_rows(inventory["tag_counts"], empty_label="_None_"),
            "",
            "## Source Projects",
            "",
            "| Source project | Count |",
            "| --- | ---: |",
            *_counter_rows(inventory["source_project_counts"], empty_label="_None_"),
            "",
            "## Content Types",
            "",
            "| Content type | Count |",
            "| --- | ---: |",
            *_counter_rows(inventory["content_type_counts"], empty_label="_None_"),
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _counter_rows(counter: Counter[str], *, empty_label: str) -> list[str]:
    if not counter:
        return [f"| {empty_label} | 0 |"]
    return [
        f"| {_markdown_cell(key)} | {count} |"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _counter_text(counter: Counter[str]) -> str:
    return ", ".join(
        f"{key} ({count})"
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    )


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, Enum):
        return "string"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list | tuple | set):
        return "array"
    return type(value).__name__


def _example_value(value: Any) -> str:
    text = _inline_text(_serializable_value(value))
    if len(text) > _MAX_EXAMPLE_LENGTH:
        text = f"{text[: _MAX_EXAMPLE_LENGTH - 1].rstrip()}..."
    return text


def _serializable_value(value: Any) -> str:
    normalized = _normalized_value(value)
    if isinstance(normalized, str):
        return normalized
    if normalized is None or isinstance(normalized, int | float | bool):
        return str(normalized).lower() if isinstance(normalized, bool) else str(normalized)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str)


def _normalized_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return _normalized_value(value.model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): _normalized_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, list | tuple):
        return [_normalized_value(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalized_value(item) for item in value), key=str)
    return value


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (
        _field_value(unit.source_project),
        _inline_text(unit.source_id),
        _inline_text(unit.title),
        _inline_text(unit.id),
    )


def _path_part(value: Any) -> str:
    return _inline_text(value).replace(".", "\\.")


def _field_value(value: Any) -> str:
    return _inline_text(getattr(value, "value", value))


def _inline_text(value: Any) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: Any) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")
