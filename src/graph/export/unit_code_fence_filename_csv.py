"""CSV export for code fence filename attributes."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line", "language", "filename", "attribute_name"]
_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*(.*?)\s*$")
_ATTRS = {"filename", "file", "title"}


def export_unit_code_fence_filename_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line"]), sort_key(row["attribute_name"]), sort_key(row["filename"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        match = _FENCE_RE.match(line)
        if not match:
            continue
        info = match.group(2)
        language = _language(info)
        for name, value in _attributes(info):
            if name in _ATTRS and value:
                rows.append({"unit_id": uid, "title": title, "line": line_number, "language": language, "filename": value, "attribute_name": name})
    return rows


def _language(info: str) -> str:
    token = shlex.split(info)[0] if info.strip() else ""
    return "" if "=" in token else field_value(token)


def _attributes(info: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for token in shlex.split(info):
        if "=" not in token:
            continue
        name, value = token.split("=", 1)
        pairs.append((name.casefold(), field_value(value)))
    return pairs
