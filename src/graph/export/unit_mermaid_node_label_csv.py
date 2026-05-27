"""CSV export for Mermaid node labels in unit code fences."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "fence_start_line", "node_id", "label", "shape_hint"]
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_-]+)?")
_NODE_RE = re.compile(r"\b(?P<id>[A-Za-z][\w-]*)\s*(?P<form>\[[^\]]+\]|\([^)]+\)|\{[^}]+\})")


def export_units_to_mermaid_node_label_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows: list[dict[str, str | int]] = []
    for unit in unit_list:
        rows.extend(_rows(unit))
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["fence_start_line"]), sort_key(row["node_id"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    in_mermaid = False
    fence_start = 0
    seen: set[str] = set()
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        fence = _FENCE_RE.match(line)
        if fence:
            if in_mermaid:
                in_mermaid = False
                seen = set()
            else:
                in_mermaid = (fence.group(2) or "").casefold() == "mermaid"
                fence_start = line_number
                seen = set()
            continue
        if not in_mermaid:
            continue
        for match in _NODE_RE.finditer(line):
            node_id = field_value(match.group("id"))
            if node_id in seen:
                continue
            seen.add(node_id)
            label, shape = _label_and_shape(match.group("form"))
            rows.append({"unit_id": unit_id(unit), "fence_start_line": fence_start, "node_id": node_id, "label": label, "shape_hint": shape})
    return rows


def _label_and_shape(form: str) -> tuple[str, str]:
    if form.startswith("["):
        inner = form[1:-1].strip()
        return _unquote(inner), "quoted" if _is_quoted(inner) else "bracket"
    if form.startswith("("):
        return field_value(form[1:-1]), "paren"
    return field_value(form[1:-1]), "brace"


def _is_quoted(value: str) -> bool:
    return len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]


def _unquote(value: str) -> str:
    return field_value(value[1:-1] if _is_quoted(value) else value)
