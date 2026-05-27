"""Summarize inline Markdown link titles in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\((\S+?)(?:\s+((?:\"[^\"]*\"|'[^']*'|\([^)]*\))))?\)")


def summarize_unit_markdown_link_titles(units: Iterable[Any], sample_limit: int = 10) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = link_count = titled = 0
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        found = False
        for line_number, line in _content_lines(str(get(unit, "content") or "")):
            for match in _LINK_RE.finditer(line):
                text, url, title = match.groups()
                if url.startswith("["):
                    continue
                found = True
                link_count += 1
                clean_title = _clean_title(title or "")
                if clean_title:
                    titled += 1
                if len(samples) < limit:
                    samples.append({"unit_id": uid, "line_number": line_number, "text": text, "url": url, "link_title": clean_title})
        if found:
            units_with += 1
    return {"total_units": total, "units_with_link_titles": units_with, "markdown_link_count": link_count, "titled_link_count": titled, "untitled_link_count": link_count - titled, "link_title_samples": samples}


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


def _clean_title(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and ((text[0] == text[-1] and text[0] in {"'", '"'}) or (text[0] == "(" and text[-1] == ")")):
        return text[1:-1]
    return text
