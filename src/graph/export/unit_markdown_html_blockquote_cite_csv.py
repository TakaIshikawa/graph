"""CSV export for Markdown-embedded HTML blockquote cite attributes."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "cite_url", "domain", "text_preview", "multiline"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_BLOCKQUOTE_RE = re.compile(r"<blockquote\b(?P<attrs>[^>]*)>(?P<body>.*?)</blockquote\s*>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""")
_TAG_RE = re.compile(r"<[^>]+>")
_PATH_KEYS = ("path", "source_path", "file_path", "filename", "source_url")
_SOURCE_KEYS = ("source", "source_name", "source_id")


def export_units_to_markdown_html_blockquote_cite_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["cite_url"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    content = _without_fences(str(get(unit, "content") or data.get("content") or ""))
    rows: list[dict[str, str | int]] = []
    for match in _BLOCKQUOTE_RE.finditer(content):
        attrs = _attrs(match.group("attrs"))
        cite = attrs.get("cite", "")
        if not cite:
            continue
        rows.append(
            {
                "unit_id": unit_id(unit),
                "title": field_value(get(unit, "title") or data.get("title")),
                "source_path": _first_value(unit, data, _PATH_KEYS),
                "source": _first_value(unit, data, _SOURCE_KEYS),
                "line_number": content.count("\n", 0, match.start()) + 1,
                "cite_url": cite,
                "domain": _domain(cite),
                "text_preview": _preview(match.group("body")),
                "multiline": str("\n" in match.group(0)).lower(),
            }
        )
    return rows


def _without_fences(content: str) -> str:
    lines: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            lines.append("")
            continue
        lines.append("" if in_fence else line)
    return "\n".join(lines)


def _attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw):
        attrs[match.group(1).casefold()] = field_value(html.unescape(match.group(2) or match.group(3) or match.group(4)))
    return attrs


def _domain(url: str) -> str:
    parsed = urlparse(url)
    return parsed.hostname or ""


def _preview(raw: str, limit: int = 120) -> str:
    text = field_value(html.unescape(_TAG_RE.sub(" ", raw)))
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}..."


def _first_value(unit: Mapping[str, Any] | object, data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = field_value(get(unit, key)) or field_value(data.get(key))
        if text:
            return text
    return ""
