"""CSV export for Markdown and HTML link rel attributes in unit content."""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "link_text", "href", "rel_value", "rel_token_count", "nofollow", "noopener", "noreferrer", "line_number"]
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[([^\]\n]+)\]\(([^)\s]+)(?:\s+[^)]*)?\)\s*\{([^}]*)\}")
_ANCHOR_RE = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<text>.*?)</a>", re.IGNORECASE)
_ATTR_RE = re.compile(r"(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)", re.DOTALL)


def export_units_to_markdown_link_rel_attribute_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["href"]), sort_key(row["link_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int | bool]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    rows: list[dict[str, str | int | bool]] = []
    for line_number, line in _content_lines(str(get(unit, "content") or data.get("content") or "")):
        for match in _MARKDOWN_LINK_RE.finditer(line):
            rel_value = _rel_from_markdown_attrs(match.group(3))
            if rel_value:
                rows.append(_row(uid, title, match.group(1), match.group(2), rel_value, line_number))
        for match in _ANCHOR_RE.finditer(line):
            attrs = {m.group("name").casefold(): field_value(m.group("value")) for m in _ATTR_RE.finditer(match.group("attrs"))}
            if attrs.get("href") and attrs.get("rel"):
                rows.append(_row(uid, title, re.sub(r"<[^>]+>", "", match.group("text")), attrs["href"], attrs["rel"], line_number))
    return rows


def _row(uid: str, title: str, link_text: str, href: str, rel_value: str, line_number: int) -> dict[str, str | int | bool]:
    tokens = [token.casefold() for token in rel_value.split()]
    return {
        "unit_id": uid,
        "title": title,
        "link_text": field_value(link_text),
        "href": field_value(href),
        "rel_value": field_value(rel_value),
        "rel_token_count": len(tokens),
        "nofollow": "nofollow" in tokens,
        "noopener": "noopener" in tokens,
        "noreferrer": "noreferrer" in tokens,
        "line_number": line_number,
    }


def _rel_from_markdown_attrs(attrs: str) -> str:
    for token in shlex.split(attrs, posix=True):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key.casefold() == "rel":
            return field_value(value.strip("\"'"))
    return ""


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
