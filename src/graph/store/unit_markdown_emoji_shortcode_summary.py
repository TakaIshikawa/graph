"""Summarize Markdown emoji shortcodes in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_CODE_SPAN_RE = re.compile(r"`+[^`]*`+")
_VALID_RE = re.compile(r"(?<![\w/:]):([A-Za-z0-9][A-Za-z0-9_-]*):(?![\w/])")
_MALFORMED_RE = re.compile(r"(?<![\w/]):([A-Za-z0-9_-]*[^A-Za-z0-9_:\s][^\s:]*|[A-Za-z0-9_-]{0,1}):(?![\w/])")


def summarize_unit_markdown_emoji_shortcodes(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = 0
    counts: Counter[str] = Counter()
    unit_sets: dict[str, set[str]] = {}
    samples: list[dict[str, str | int]] = []
    malformed: list[dict[str, str | int]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        for line_number, line in _content_lines(str(get(unit, "content") or "")):
            masked = _CODE_SPAN_RE.sub(lambda match: " " * (match.end() - match.start()), line)
            for match in _VALID_RE.finditer(masked):
                name = match.group(1).casefold()
                counts[name] += 1
                unit_sets.setdefault(name, set()).add(uid)
                if len(samples) < limit:
                    samples.append({"unit_id": uid, "line_number": line_number, "shortcode": f":{name}:"})
            valid_spans = [range(match.start(), match.end()) for match in _VALID_RE.finditer(masked)]
            for match in _MALFORMED_RE.finditer(masked):
                if any(match.start() in span for span in valid_spans):
                    continue
                if len(malformed) < limit:
                    malformed.append({"unit_id": uid, "line_number": line_number, "token": field_value(match.group(0))})
    shortcodes = [{"shortcode": key, "count": count, "unit_count": len(unit_sets[key])} for key, count in counts.items()]
    shortcodes.sort(key=lambda row: (-int(row["count"]), sort_key(row["shortcode"])))
    return {"total_units": total, "units_with_shortcodes": len({uid for values in unit_sets.values() for uid in values}), "shortcodes": shortcodes, "malformed_tokens": malformed, "samples": samples}


def _content_lines(content: str) -> list[tuple[int, str]]:
    rows = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_number, line))
    return rows
