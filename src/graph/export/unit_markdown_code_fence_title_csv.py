"""CSV export for Markdown code fences with title-like attributes."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "start_line", "end_line", "language", "title_attribute", "attribute_name"]
_FENCE_RE = re.compile(r"^\s*(?P<fence>`{3,}|~{3,})\s*(?P<info>.*)$")
_ATTR_NAMES = {"title", "name", "filename"}
_KEY_VALUE_RE = re.compile(r"^(?P<key>[A-Za-z_][\w-]*)=(?P<value>.*)$")


def export_unit_markdown_code_fence_title_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["start_line"]), sort_key(row["attribute_name"]), sort_key(row["title_attribute"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    lines = str(get(unit, "content") or data.get("content") or "").splitlines()
    rows: list[dict[str, str | int]] = []
    active: list[dict[str, str | int]] | None = None
    fence_marker = ""
    for line_number, line in enumerate(lines, start=1):
        match = _FENCE_RE.match(line)
        if not match:
            continue
        marker = match.group("fence")[0]
        if active is not None:
            if marker == fence_marker:
                for row in active:
                    row["end_line"] = line_number
                rows.extend(active)
                active = None
                fence_marker = ""
            continue
        language, attrs = _parse_info(match.group("info"))
        if not attrs:
            fence_marker = marker
            active = []
            continue
        fence_marker = marker
        active = [
            {"unit_id": uid, "title": title, "source": source, "start_line": line_number, "end_line": len(lines), "language": language, "title_attribute": value, "attribute_name": name}
            for name, value in attrs
        ]
    if active:
        rows.extend(active)
    return rows


def _parse_info(info: str) -> tuple[str, list[tuple[str, str]]]:
    try:
        tokens = shlex.split(info, posix=True)
    except ValueError:
        tokens = info.split()
    language = field_value(tokens[0]) if tokens and "=" not in tokens[0] and not tokens[0].startswith((".", "#", "{")) else ""
    attrs: list[tuple[str, str]] = []
    for token in tokens[1 if language else 0 :]:
        token = token.strip("{}")
        match = _KEY_VALUE_RE.match(token)
        if match and match.group("key").casefold() in _ATTR_NAMES:
            attrs.append((field_value(match.group("key")), field_value(match.group("value").strip("\"'"))))
    return language, attrs
