"""CSV export for Markdown-embedded HTML sup and sub spans."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = ["unit_id", "title", "source_path", "source", "line_number", "tag_name", "span_text", "is_reference_like", "context_preview"]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_SPAN_RE = re.compile(r"<(?P<tag>sup|sub)\b[^>]*>(?P<body>.*?)</(?P=tag)\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_PATH_KEYS = ("path", "source_path", "file_path", "filename", "source_url")
_SOURCE_KEYS = ("source", "source_name", "source_id")


def export_units_to_markdown_html_sup_sub_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["tag_name"]), sort_key(row["span_text"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    rows: list[dict[str, str | int]] = []
    for line_number, line in _content_lines(str(get(unit, "content") or data.get("content") or "")):
        for match in _SPAN_RE.finditer(line):
            tag = match.group("tag").casefold()
            text = _text(match.group("body"))
            rows.append(
                {
                    "unit_id": unit_id(unit),
                    "title": field_value(get(unit, "title") or data.get("title")),
                    "source_path": _first_value(unit, data, _PATH_KEYS),
                    "source": _first_value(unit, data, _SOURCE_KEYS),
                    "line_number": line_number,
                    "tag_name": tag,
                    "span_text": text,
                    "is_reference_like": str(tag == "sup" and _reference_like(text)).lower(),
                    "context_preview": _preview(line),
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


def _text(raw: str) -> str:
    return field_value(html.unescape(_TAG_RE.sub(" ", raw)))


def _reference_like(text: str) -> bool:
    return bool(re.fullmatch(r"(?:\d{1,3}|[a-z]|[ivxlcdm]{1,8}|\[[^\]]{1,20}\])", text.casefold()))


def _preview(raw: str, limit: int = 120) -> str:
    text = field_value(html.unescape(_TAG_RE.sub("", raw)))
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}..."


def _first_value(unit: Mapping[str, Any] | object, data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = field_value(get(unit, key)) or field_value(data.get(key))
        if text:
            return text
    return ""
