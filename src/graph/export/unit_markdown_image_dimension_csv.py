"""CSV export for Markdown images with dimensions."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "alt_text", "target", "width", "height", "dimension_source"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_IMAGE_RE = re.compile(r"!\[([^\]\n]*)\]\(([^)\s]+)(?:\s+[^)]*)?\)(?:\s*\{([^}]*)\})?")


def export_units_to_markdown_image_dimension_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or metadata(unit).get("title"))
    rows: list[dict[str, str | int]] = []
    for line_number, line in _content_lines(str(get(unit, "content") or "")):
        for match in _IMAGE_RE.finditer(line):
            target = field_value(match.group(2))
            width, height, source = _dimensions(target, match.group(3) or "")
            if not width and not height:
                continue
            rows.append({"unit_id": uid, "title": title, "line_number": line_number, "alt_text": field_value(match.group(1)), "target": target, "width": width, "height": height, "dimension_source": source})
    return rows


def _dimensions(target: str, attrs: str) -> tuple[str, str, str]:
    values = {}
    for token in shlex.split(attrs, posix=True) if attrs.strip() else []:
        if "=" in token:
            key, value = token.split("=", 1)
            values[key.lower()] = value.strip("\"'")
    if "width" in values or "height" in values:
        return field_value(values.get("width", "")), field_value(values.get("height", "")), "attribute"
    split = urlsplit(target)
    params = {key.lower(): vals[-1] for key, vals in parse_qs(split.query).items() if vals}
    params.update({key.lower(): vals[-1] for key, vals in parse_qs(split.fragment).items() if vals})
    if "width" in params or "height" in params:
        return field_value(params.get("width", "")), field_value(params.get("height", "")), "url"
    return "", "", ""


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
