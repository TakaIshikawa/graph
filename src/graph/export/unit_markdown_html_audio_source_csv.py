"""CSV export for Markdown-embedded HTML audio and source elements."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from graph.export._report_csv import field_value, get, metadata, render_csv, sort_key, unit_id, write_csv

_FIELDNAMES = [
    "unit_id",
    "title",
    "source_path",
    "source",
    "line_number",
    "audio_src",
    "source_src",
    "type",
    "controls",
    "autoplay",
    "loop",
    "preload",
    "fallback_text",
]
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_AUDIO_RE = re.compile(r"<audio\b(?P<attrs>[^>]*)>(?P<body>.*?)</audio\s*>|<audio\b(?P<single_attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_SOURCE_RE = re.compile(r"<source\b(?P<attrs>[^>]*)/?>", re.IGNORECASE)
_ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""")
_TAG_RE = re.compile(r"<[^>]+>")
_PATH_KEYS = ("path", "source_path", "file_path", "filename", "source_url")
_SOURCE_KEYS = ("source", "source_name", "source_id")


def export_units_to_markdown_html_audio_source_csv(
    units: Iterable[Mapping[str, Any] | object], path: str | Path | None = None
) -> str | dict[str, Any]:
    unit_list = list(units)
    rows = [row for unit in unit_list for row in _rows(unit)]
    rows.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["audio_src"]), sort_key(row["source_src"])))
    text = render_csv(rows, _FIELDNAMES)
    if path is None:
        return text
    output_path, bytes_written = write_csv(path, text)
    return {"path": output_path, "unit_count": len(unit_list), "rows_exported": len(rows), "bytes_written": bytes_written}


def _rows(unit: Mapping[str, Any] | object) -> list[dict[str, str | int]]:
    data = metadata(unit)
    context = {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title") or data.get("title")),
        "source_path": _first_value(unit, data, _PATH_KEYS),
        "source": _first_value(unit, data, _SOURCE_KEYS),
    }
    content = _without_fences(str(get(unit, "content") or data.get("content") or ""))
    rows: list[dict[str, str | int]] = []
    for match in _AUDIO_RE.finditer(content):
        attrs = _attrs(match.group("attrs") or match.group("single_attrs") or "")
        body = match.group("body") or ""
        line_number = content.count("\n", 0, match.start()) + 1
        fallback = _preview(_SOURCE_RE.sub(" ", body))
        source_matches = list(_SOURCE_RE.finditer(body))
        if attrs.get("src", ""):
            rows.append(_row(context, line_number, attrs, attrs.get("src", ""), "", attrs.get("type", ""), fallback))
        for source_match in source_matches:
            source_attrs = _attrs(source_match.group("attrs"))
            source_line = line_number + body.count("\n", 0, source_match.start())
            rows.append(
                _row(
                    context,
                    source_line,
                    attrs,
                    attrs.get("src", ""),
                    source_attrs.get("src", ""),
                    source_attrs.get("type", ""),
                    fallback,
                )
            )
    return rows


def _row(
    context: dict[str, str], line_number: int, audio_attrs: dict[str, str], audio_src: str, source_src: str, mime_type: str, fallback: str
) -> dict[str, str | int]:
    return {
        **context,
        "line_number": line_number,
        "audio_src": audio_src,
        "source_src": source_src,
        "type": mime_type,
        "controls": _bool_attr(audio_attrs, "controls"),
        "autoplay": _bool_attr(audio_attrs, "autoplay"),
        "loop": _bool_attr(audio_attrs, "loop"),
        "preload": audio_attrs.get("preload", ""),
        "fallback_text": fallback,
    }


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
        value = next((group for group in match.groups()[1:] if group is not None), "")
        attrs[match.group(1).casefold()] = field_value(html.unescape(value))
    return attrs


def _bool_attr(attrs: dict[str, str], key: str) -> str:
    return str(key in attrs).lower()


def _preview(raw: str, limit: int = 120) -> str:
    text = field_value(html.unescape(_TAG_RE.sub(" ", raw)))
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}..."


def _first_value(unit: Mapping[str, Any] | object, data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = field_value(get(unit, key)) or field_value(data.get(key))
        if text:
            return text
    return ""
