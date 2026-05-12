"""Markdown export for metadata value frequencies by key."""

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


def export_metadata_value_frequency_markdown(
    units: Iterable[KnowledgeUnit],
    path: str | Path | None = None,
    *,
    keys: Iterable[str] | None = None,
    top_values: int = 5,
    min_count: int = 1,
) -> str | dict[str, Any]:
    """Return or write a deterministic Markdown report of scalar metadata value counts."""
    if not isinstance(top_values, int) or isinstance(top_values, bool) or top_values < 1:
        raise ValueError("top_values must be a positive integer")
    if not isinstance(min_count, int) or isinstance(min_count, bool) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    normalized_keys = _normalize_keys(keys)

    unit_list = sorted(list(units), key=_unit_sort_key)
    sections = _frequency_sections(unit_list, keys=normalized_keys, top_values=top_values, min_count=min_count)
    text = _render_report(
        sections,
        units_scanned=len(unit_list),
        keys=normalized_keys,
        top_values=top_values,
        min_count=min_count,
    )

    if path is None:
        return text

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return {
        "path": str(output_path),
        "units_scanned": len(unit_list),
        "keys_requested": None if normalized_keys is None else list(normalized_keys),
        "keys_exported": len(sections),
        "top_values": top_values,
        "min_count": min_count,
        "bytes_written": output_path.stat().st_size,
    }


def _frequency_sections(
    units: list[KnowledgeUnit],
    *,
    keys: tuple[str, ...] | None,
    top_values: int,
    min_count: int,
) -> list[dict[str, Any]]:
    counts: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    payloads: dict[tuple[str, str], Any] = {}
    key_filter = set(keys) if keys is not None else None

    for unit in units:
        if not isinstance(unit.metadata, Mapping):
            continue
        for raw_key, raw_value in unit.metadata.items():
            key = _inline_text(raw_key)
            if not key or (key_filter is not None and key not in key_filter):
                continue
            for value in _scalar_values(raw_value):
                value_key = _value_key(value)
                counts[key][value_key] += 1
                payloads.setdefault(value_key, _normalized_value(value))

    candidate_keys = keys if keys is not None else tuple(sorted(counts, key=_sort_key))
    sections: list[dict[str, Any]] = []
    for key in candidate_keys:
        rows = [
            {
                "value": _value_text(payloads[value_key]),
                "count": count,
            }
            for value_key, count in sorted(
                counts.get(key, Counter()).items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
            if count >= min_count
        ][:top_values]
        if rows:
            sections.append({"key": key, "rows": rows})
    return sections


def _normalize_keys(keys: Iterable[str] | None) -> tuple[str, ...] | None:
    if keys is None:
        return None
    normalized = tuple(sorted({_inline_text(key) for key in keys if _inline_text(key)}, key=_sort_key))
    if not normalized:
        raise ValueError("keys must contain at least one non-empty key or be None")
    return normalized


def _scalar_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        return
    if isinstance(value, list | tuple | set):
        for child in sorted(value, key=lambda item: _value_key(item)):
            if _is_scalar(child):
                yield child
        return
    if _is_scalar(value):
        yield value


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool | Enum | datetime | date)


def _value_key(value: Any) -> tuple[str, str]:
    normalized = _normalized_value(value)
    return (_value_type(value), json.dumps(normalized, sort_keys=True, ensure_ascii=False, default=str))


def _value_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _normalized_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _render_report(
    sections: list[dict[str, Any]],
    *,
    units_scanned: int,
    keys: tuple[str, ...] | None,
    top_values: int,
    min_count: int,
) -> str:
    lines = [
        "# Metadata Value Frequency",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Units scanned | {units_scanned} |",
        f"| Keys reported | {len(sections)} |",
        f"| Keys filter | {_markdown_cell(_keys_text(keys))} |",
        f"| Top values | {top_values} |",
        f"| Min count | {min_count} |",
        "",
        "## Values",
    ]
    if sections:
        for section in sections:
            lines.extend(
                [
                    "",
                    f"### {_heading_text(section['key'])}",
                    "",
                    "| Value | Count |",
                    "| --- | ---: |",
                ]
            )
            for row in section["rows"]:
                lines.append(f"| {_markdown_cell(row['value'])} | {row['count']} |")
    else:
        lines.extend(["", "| Metadata key | Value | Count |", "| --- | --- | ---: |", "| _None_ | _None_ | 0 |"])
    return "\n".join(lines).rstrip() + "\n"


def _keys_text(keys: tuple[str, ...] | None) -> str:
    return "_All_" if keys is None else ", ".join(keys)


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


def _heading_text(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("#", "\\#")


def _unit_sort_key(unit: KnowledgeUnit) -> tuple[str, str, str, str]:
    return (_unit_source(unit), _inline_text(unit.source_id), _inline_text(unit.title), _inline_text(unit.id))


def _unit_source(unit: KnowledgeUnit) -> str:
    return _inline_text(getattr(unit.source_project, "value", unit.source_project)) or "Unknown"


def _inline_text(value: object) -> str:
    text = "" if value is None else str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _markdown_cell(value: object) -> str:
    return _inline_text(value).replace("\\", "\\\\").replace("|", "\\|")


def _sort_key(value: object) -> tuple[str, str]:
    text = _inline_text(value)
    return (text.casefold(), text)
