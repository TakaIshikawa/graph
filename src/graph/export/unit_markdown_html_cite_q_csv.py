"""CSV export for Markdown-embedded HTML cite and q elements."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "line_number", "tag", "text", "cite_url", "raw_html"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_TAG_RE = re.compile(r"<(?P<tag>cite|q)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)\s*>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+))""")
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def export_unit_markdown_html_cite_q_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag"]), sort_key(row["text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    uid = unit_id(unit)
    title = field_value(get(unit, "title") or data.get("title"))
    rows: list[dict[str, str | int]] = []
    in_fence = False

    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _TAG_RE.finditer(line):
            tag = match.group("tag").lower()
            attrs = _attrs(match.group("attrs"))
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "line_number": line_number,
                    "tag": tag,
                    "text": field_value(html.unescape(_HTML_TAG_RE.sub(" ", match.group("body")))),
                    "cite_url": attrs.get("cite", ""),
                    "raw_html": match.group(0),
                }
            )
    return rows


def _attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in _ATTR_RE.finditer(raw):
        attrs[match.group(1).casefold()] = field_value(match.group(2) or match.group(3) or match.group(4))
    return attrs
