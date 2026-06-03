"""CSV export for Markdown-embedded HTML code and pre elements."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source", "line_number", "tag", "text", "class", "language_hint"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_ELEMENT_RE = re.compile(r"<(?P<tag>code|pre)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_CLASS_RE = re.compile(r"\bclass\s*=\s*(['\"])(?P<value>.*?)\1", re.IGNORECASE)


def export_unit_markdown_html_code_pre_csv(units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None) -> str | dict[str, Any]:
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
    source = field_value(get(unit, "source") or get(unit, "source_url") or data.get("source") or data.get("source_url"))
    rows: list[dict[str, str | int]] = []
    in_fence = False
    for line_number, line in enumerate(str(get(unit, "content") or data.get("content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in _ELEMENT_RE.finditer(line):
            class_value = _class_value(match.group("attrs"))
            rows.append(
                {
                    "unit_id": uid,
                    "title": title,
                    "source": source,
                    "line_number": line_number,
                    "tag": match.group("tag").casefold(),
                    "text": field_value(html.unescape(_TAG_RE.sub("", match.group("body")))),
                    "class": class_value,
                    "language_hint": _language_hint(class_value),
                }
            )
    return rows


def _class_value(attrs: str) -> str:
    match = _CLASS_RE.search(attrs)
    return field_value(match.group("value") if match else "")


def _language_hint(class_value: str) -> str:
    for token in class_value.split():
        if token.casefold().startswith("language-") and len(token) > len("language-"):
            return token[len("language-") :]
    return ""
