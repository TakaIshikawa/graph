"""Summarize aliased Obsidian-style Markdown wikilinks in units."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_WIKILINK_RE = re.compile(r"(?<!!)\[\[([^\[\]\n]+)\]\]")


def summarize_unit_markdown_wikilink_aliases(units: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize wikilink aliases such as ``[[target|alias]]``."""
    unit_list = list(units)
    total_wikilinks = aliased_wikilinks = 0
    units_with_aliases: set[str] = set()
    pair_counts: Counter[tuple[str, str]] = Counter()
    alias_counts: Counter[str] = Counter()
    samples: list[dict[str, str | int]] = []
    for index, unit in enumerate(unit_list):
        uid = unit_id(unit) or str(index)
        unit_has_alias = False
        for line_number, raw in _wikilinks(_content(unit)):
            total_wikilinks += 1
            if "|" not in raw:
                continue
            target, alias = (field_value(part).strip() for part in raw.split("|", 1))
            if not target or not alias:
                continue
            aliased_wikilinks += 1
            unit_has_alias = True
            pair_counts[(alias, target)] += 1
            alias_counts[alias] += 1
            samples.append({"unit_id": uid, "target": target, "alias": alias, "line_number": line_number})
        if unit_has_alias:
            units_with_aliases.add(uid)
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["target"]), sort_key(row["alias"])))
    return {
        "total_units": len(unit_list),
        "total_wikilinks": total_wikilinks,
        "aliased_wikilinks": aliased_wikilinks,
        "units_with_aliases": len(units_with_aliases),
        "alias_target_pairs": [
            {"alias": alias, "target": target, "count": count}
            for (alias, target), count in sorted(pair_counts.items(), key=lambda item: (sort_key(item[0][0]), sort_key(item[0][1])))
        ],
        "top_aliases": [{"alias": alias, "count": count} for alias, count in sorted(alias_counts.items(), key=lambda item: (-item[1], sort_key(item[0])))],
        "samples": samples[:sample_limit],
    }


def _content(unit: Mapping[str, Any] | object) -> str:
    return str(get(unit, "content") or metadata(unit).get("content") or "")


def _wikilinks(content: str) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    in_fence = False
    for line_number, line in enumerate(content.splitlines(), start=1):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            rows.extend((line_number, match.group(1)) for match in _WIKILINK_RE.finditer(line))
    return rows
