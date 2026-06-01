"""Summarize HTML comments in unit Markdown content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_COMMENT_RE = re.compile(r"<!--(.*?)-->", re.DOTALL)
_KEYWORD_RE = re.compile(r"\b(TODO|FIXME|NOTE)\b", re.IGNORECASE)


def summarize_unit_markdown_html_comments(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    unit_list = list(units)
    rows: list[dict[str, Any]] = []
    total_comments = multiline_comments = keyword_comments = 0
    for unit in unit_list:
        comments = [_comment(match.group(1)) for match in _COMMENT_RE.finditer(str(get(unit, "content") or ""))]
        if not comments:
            continue
        total_comments += len(comments)
        multiline = sum(1 for comment in comments if "\n" in comment)
        keyword = sum(1 for comment in comments if _KEYWORD_RE.search(comment))
        multiline_comments += multiline
        keyword_comments += keyword
        samples = sorted((field_value(comment) for comment in comments), key=lambda text: (-len(text), text))[:limit]
        rows.append({"unit_id": unit_id(unit), "comment_count": len(comments), "multiline_comment_count": multiline, "todo_like_comment_count": keyword, "longest_comment_samples": samples})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {"total_units": len(unit_list), "affected_unit_count": len(rows), "total_comment_count": total_comments, "multiline_comment_count": multiline_comments, "todo_like_comment_count": keyword_comments, "units": rows}


def _comment(value: str) -> str:
    return value.strip()
