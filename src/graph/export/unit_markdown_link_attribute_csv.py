"""CSV export for Pandoc-style attributes attached to Markdown links."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "link_text", "href", "attribute_name", "attribute_value", "attribute_kind"]
_LINK_RE = re.compile(r"(?<!!)\[(?P<text>[^\]\n]+)]\((?P<href>[^)\s]+)(?:\s+[^)]*)?\)\s*\{(?P<attrs>[^}]*)}")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def export_units_to_markdown_link_attribute_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["href"]), sort_key(row["attribute_name"])))
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
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _LINK_RE.finditer(line):
            if _inside_code_span(line, match.start()):
                continue
            for name, value, kind in _attributes(match.group("attrs")):
                rows.append({"unit_id": uid, "title": title, "source": source, "line_number": line_number, "link_text": field_value(match.group("text")), "href": field_value(match.group("href")), "attribute_name": name, "attribute_value": value, "attribute_kind": kind})
    return rows


def _attributes(text: str) -> list[tuple[str, str, str]]:
    try:
        tokens = shlex.split(text)
    except ValueError:
        tokens = text.split()
    attrs: list[tuple[str, str, str]] = []
    for token in tokens:
        if token.startswith("#") and len(token) > 1:
            attrs.append(("id", token[1:], "id"))
        elif token.startswith(".") and len(token) > 1:
            attrs.append(("class", token[1:], "class"))
        elif "=" in token:
            name, value = token.split("=", 1)
            attrs.append((field_value(name), field_value(value.strip("\"'")), "key_value"))
        elif token:
            attrs.append((field_value(token), "", "boolean"))
    return attrs


def _inside_code_span(line: str, offset: int) -> bool:
    return line[:offset].count("`") % 2 == 1
