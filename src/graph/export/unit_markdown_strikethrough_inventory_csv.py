"""CSV export for Markdown strikethrough spans by unit."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "strikethrough_span_count", "total_strikethrough_text_length", "empty_strikethrough_span_count", "sample_texts"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_INLINE_CODE_RE = re.compile(r"`+[^`\n]*`+")
_STRIKE_RE = re.compile(r"(?<!\\)~~(.*?)(?<!\\)~~", re.DOTALL)


def export_units_to_markdown_strikethrough_inventory_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = sorted((_row(unit) for unit in unit_list), key=lambda row: sort_key(row["unit_id"]))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _row(unit: Mapping[str, Any] | object) -> dict[str, str | int]:
    samples: list[str] = []
    lengths: list[int] = []
    empty_count = 0
    for text in _STRIKE_RE.findall(_content_without_code(unit)):
        normalized = field_value(text)
        lengths.append(len(normalized))
        if not normalized:
            empty_count += 1
        elif len(samples) < 5:
            samples.append(normalized)
    return {
        "unit_id": unit_id(unit),
        "strikethrough_span_count": len(lengths),
        "total_strikethrough_text_length": sum(lengths),
        "empty_strikethrough_span_count": empty_count,
        "sample_texts": "; ".join(samples),
    }


def _content_without_code(unit: Mapping[str, Any] | object) -> str:
    lines: list[str] = []
    in_fence = False
    for line in str(get(unit, "content") or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            lines.append(_INLINE_CODE_RE.sub("", line))
    return "\n".join(lines)
