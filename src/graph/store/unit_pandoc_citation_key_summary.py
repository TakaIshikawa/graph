"""Summarize Pandoc citation keys in unit content."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_CLUSTER_RE = re.compile(r"\[[^\]]*@[-\w:.]+[^\]]*\]")
_KEY_RE = re.compile(r"(?<![\w.])(-?)@([A-Za-z0-9][\w:.-]*)")


def summarize_unit_pandoc_citation_keys(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    total_units = mentions = clusters = suppress = 0
    key_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for index, unit in enumerate(units):
        total_units += 1
        uid = unit_id(unit) or str(index)
        for line_no, line in _content_lines(unit):
            spans: list[tuple[int, int]] = []
            for cluster in _CLUSTER_RE.finditer(line):
                clusters += 1
                spans.append(cluster.span())
                for marker, key in _KEY_RE.findall(cluster.group(0)):
                    normalized = key.rstrip(".,;:!?").casefold()
                    mentions += 1
                    suppress += 1 if marker else 0
                    key_counts[normalized] += 1
                    if len(examples[normalized]) < sample_limit:
                        examples[normalized].append({"unit_id": uid, "line": line_no, "key": normalized, "cluster": cluster.group(0)})
            for match in _KEY_RE.finditer(line):
                if any(start <= match.start() < end for start, end in spans):
                    continue
                normalized = match.group(2).rstrip(".,;:!?").casefold()
                mentions += 1
                suppress += 1 if match.group(1) else 0
                key_counts[normalized] += 1
                if len(examples[normalized]) < sample_limit:
                    examples[normalized].append({"unit_id": uid, "line": line_no, "key": normalized, "cluster": match.group(0)})
    top = [{"key": key, "count": key_counts[key], "examples": examples[key]} for key in sorted(key_counts, key=lambda k: (-key_counts[k], sort_key(k)))]
    return {"total_units": total_units, "total_citation_mentions": mentions, "unique_citation_keys": len(key_counts), "citation_cluster_count": clusters, "suppress_author_count": suppress, "top_citation_keys": top}


def _content_lines(unit: Any) -> list[tuple[int, str]]:
    in_fence = False
    rows = []
    for line_no, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.append((line_no, line))
    return rows
