"""Summarize HTML footnote backlink anchors in Markdown output."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_ANCHOR_RE = re.compile(r"<a\b(?P<attrs>[^>]*)>(?P<text>.*?)</a>", re.IGNORECASE)
_HREF_RE = re.compile(r"""href\s*=\s*(?:"([^"]+)"|'([^']+)'|([^\s>]+))""", re.IGNORECASE)


def summarize_unit_markdown_footnote_backrefs(units: Iterable[Any], sample_limit: int = 10) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = units_with = count = 0
    footnotes: Counter[str] = Counter()
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        found = False
        for line_number, line in _content_lines(str(get(unit, "content") or "")):
            for anchor in _ANCHOR_RE.finditer(line):
                href_match = _HREF_RE.search(anchor.group("attrs"))
                href = next((value for value in href_match.groups() if value is not None), "") if href_match else ""
                if not href.startswith("#fnref"):
                    continue
                footnote_id = href[6:]
                found = True
                count += 1
                footnotes[footnote_id] += 1
                if len(samples) < limit:
                    samples.append({"unit_id": uid, "line_number": line_number, "footnote_id": footnote_id, "href": href, "backref_text": field_value(re.sub(r"<[^>]+>", "", anchor.group("text")))})
        if found:
            units_with += 1
    return {"total_units": total, "units_with_footnote_backrefs": units_with, "footnote_backref_count": count, "footnote_id_counts": dict(sorted(footnotes.items())), "footnote_backref_samples": samples}


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
