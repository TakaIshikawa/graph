"""CSV export for Markdown-embedded HTML iframe elements."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "src", "title", "loading", "sandbox", "allow", "referrerpolicy", "width", "height", "domain"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_IFRAME_RE = re.compile(r"<iframe\b(?P<attrs>[^>]*)>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""")
_PATH_KEYS = ("path", "source_path", "file_path", "filename", "source_url")
_SOURCE_KEYS = ("source", "source_name", "source_id")


def export_units_to_markdown_html_iframe_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["src"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    rows: list[dict[str, str | int]] = []
    for line_number, line in _content_lines(str(get(unit, "content") or data.get("content") or "")):
        for match in _IFRAME_RE.finditer(line):
            attrs = _attrs(match.group("attrs"))
            src = attrs.get("src", "")
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "title": field_value(get(unit, "title") or data.get("title")),
                    "source_path": _first_value(unit, data, _PATH_KEYS),
                    "source": _first_value(unit, data, _SOURCE_KEYS),
                    "line_number": line_number,
                    "src": src,
                    "title": attrs.get("title", ""),
                    "loading": attrs.get("loading", ""),
                    "sandbox": attrs.get("sandbox", ""),
                    "allow": attrs.get("allow", ""),
                    "referrerpolicy": attrs.get("referrerpolicy", ""),
                    "width": attrs.get("width", ""),
                    "height": attrs.get("height", ""),
                    "domain": _domain(src),
                }
            )
    return rows


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


def _attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw):
        value = next((group for group in match.groups()[1:] if group is not None), "")
        attrs[match.group(1).casefold()] = field_value(html.unescape(value))
    return attrs


def _domain(src: str) -> str:
    parsed = urlparse(src)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return parsed.hostname or ""
    return ""


def _first_value(unit: Mapping[str, Any] | object, data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = field_value(get(unit, key)) or field_value(data.get(key))
        if text:
            return text
    return ""
