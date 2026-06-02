"""Summarize Markdown mailto links in unit content."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import unquote

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"(?<!!)\[[^\]\n]*]\((mailto:[^) \n]+)(?:\s+['\"][^)]*['\"])?\)", re.IGNORECASE)


def summarize_unit_markdown_link_mailtos(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = link_count = 0
    emails: set[str] = set()
    rows: list[dict[str, Any]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        unit_emails: list[str] = []
        for line in _content_lines(str(get(unit, "content") or "")):
            for match in _LINK_RE.finditer(line):
                link_count += 1
                email = _email(match.group(1))
                if email:
                    emails.add(email)
                    unit_emails.append(email)
        if unit_emails:
            units_with += 1
            rows.append({"unit_id": uid, "mailto_link_count": len(unit_emails), "unique_email_count": len(set(unit_emails)), "sample_emails": sorted(set(unit_emails), key=sort_key)[:limit]})
    rows.sort(key=lambda row: sort_key(row["unit_id"]))
    return {
        "total_units": total,
        "units_with_mailto_links": units_with,
        "mailto_link_count": link_count,
        "unique_email_count": len(emails),
        "samples": sorted(emails, key=sort_key)[:limit],
        "units": rows,
    }


def _email(url: str) -> str:
    text = url[len("mailto:") :]
    text = text.split("?", 1)[0]
    return field_value(unquote(text)).casefold()


def _content_lines(content: str) -> list[str]:
    rows: list[str] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append(line)
    return rows
