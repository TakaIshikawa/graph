"""Shared helpers for Markdown-embedded HTML CSV exporters."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from graph.export._report_csv import field_value, get, metadata, unit_id

FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
ATTR_RE = re.compile(r"""([A-Za-z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""")
TAG_RE = re.compile(r"<[^>]+>")
PATH_KEYS = ("path", "source_path", "file_path", "filename", "source_url")
SOURCE_KEYS = ("source", "source_name", "source_id")


def unit_context(unit: Mapping[str, Any] | object) -> dict[str, str]:
    data = metadata(unit)
    return {
        "unit_id": unit_id(unit),
        "title": field_value(get(unit, "title") or data.get("title")),
        "source_path": first_value(unit, data, PATH_KEYS),
        "source": first_value(unit, data, SOURCE_KEYS),
    }


def content_without_fences(unit: Mapping[str, Any] | object) -> str:
    data = metadata(unit)
    content = str(get(unit, "content") or data.get("content") or "")
    lines: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            lines.append("")
            continue
        lines.append("" if in_fence else line)
    return "\n".join(lines)


def attrs(raw: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for match in ATTR_RE.finditer(raw):
        value = next((group for group in match.groups()[1:] if group is not None), "")
        values[match.group(1).casefold()] = field_value(html.unescape(value))
    return values


def bool_attr(values: Mapping[str, str], key: str) -> str:
    return str(key in values).lower()


def has_attr(values: Mapping[str, str], key: str) -> str:
    return bool_attr(values, key)


def preview(raw: str, limit: int = 120) -> str:
    text = field_value(html.unescape(TAG_RE.sub(" ", raw)))
    return text if len(text) <= limit else f"{text[: limit - 1].rstrip()}..."


def line_number(content: str, index: int) -> int:
    return content.count("\n", 0, index) + 1


def domain(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return parsed.hostname or ""
    return ""


def first_value(unit: Mapping[str, Any] | object, data: Mapping[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        text = field_value(get(unit, key)) or field_value(data.get(key))
        if text:
            return text
    return ""
