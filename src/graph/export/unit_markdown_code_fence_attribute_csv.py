"""CSV export for Markdown code fence info-string attributes."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line", "language", "attribute_type", "attribute_name", "attribute_value"]
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*(?P<info>.*)$")
_BRACE_RE = re.compile(r"\{(?P<body>[^}]*)\}")
_KEY_VALUE_RE = re.compile(r"^(?P<key>[A-Za-z_][\w-]*)=(?P<value>.*)$")


def export_units_to_markdown_code_fence_attribute_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["attribute_type"]), sort_key(row["attribute_name"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        match = _FENCE_RE.match(line)
        if not match:
            continue
        if in_fence:
            in_fence = False
            continue
        in_fence = True
        language, attrs = _parse_info(match.group("info"))
        for attr_type, name, value in attrs:
            rows.append({"unit_id": uid, "title": title, "line": line_number, "language": language, "attribute_type": attr_type, "attribute_name": name, "attribute_value": value})
    return rows


def _parse_info(info: str) -> tuple[str, list[tuple[str, str, str]]]:
    brace_bodies = _BRACE_RE.findall(info)
    remaining = _BRACE_RE.sub(" ", info)
    tokens = shlex.split(remaining, posix=True) if remaining.strip() else []
    language = field_value(tokens[0]) if tokens and not tokens[0].startswith((".", "#")) and "=" not in tokens[0] else ""
    attrs: list[tuple[str, str, str]] = []
    for body in brace_bodies:
        for token in shlex.split(body, posix=True):
            attrs.extend(_attr(token))
    for token in tokens[1 if language else 0 :]:
        attrs.extend(_attr(token))
    return language, attrs


def _attr(token: str) -> list[tuple[str, str, str]]:
    if token.startswith("#") and len(token) > 1:
        return [("id", field_value(token[1:]), "")]
    if token.startswith(".") and len(token) > 1:
        return [("class", field_value(token[1:]), "")]
    match = _KEY_VALUE_RE.match(token)
    if match:
        return [("key_value", field_value(match.group("key")), field_value(match.group("value").strip("\"'")))]
    return []
