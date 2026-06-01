"""Summarize duplicate Markdown reference definitions."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_DEF_RE = re.compile(r"^[ \t]{0,3}\[([^\]\n]+)]\s*:\s*(\S.*)?$")


def summarize_unit_markdown_reference_definition_duplicates(units: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize duplicate reference definition labels within each unit."""
    limit = max(0, sample_limit)
    total = duplicate_labels = 0
    affected: set[str] = set()
    examples: list[dict[str, str | int | list[str]]] = []
    for unit in units:
        total += 1
        uid = unit_id(unit)
        labels: dict[str, list[str]] = defaultdict(list)
        display: dict[str, str] = {}
        for label, target in _definitions(str(get(unit, "content") or "")):
            normalized = _normalize_label(label)
            display.setdefault(normalized, normalized)
            labels[normalized].append(target)
        for label, targets in labels.items():
            if len(targets) < 2:
                continue
            duplicate_labels += 1
            affected.add(uid)
            examples.append(
                {
                    "unit_id": uid,
                    "label": display[label],
                    "occurrence_count": len(targets),
                    "first_target": targets[0],
                    "duplicate_targets": targets[1:],
                }
            )
    examples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["label"])))
    return {
        "total_units": total,
        "duplicate_label_count": duplicate_labels,
        "affected_units": len(affected),
        "examples": examples[:limit],
    }


def _definitions(content: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    in_fence = False
    for line in content.splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _DEF_RE.match(line)
        if match:
            rows.append((field_value(match.group(1)), field_value(match.group(2) or "")))
    return rows


def _normalize_label(value: str) -> str:
    return re.sub(r"\s+", " ", field_value(value)).casefold()
