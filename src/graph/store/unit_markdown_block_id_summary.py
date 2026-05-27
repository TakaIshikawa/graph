"""Summarize Obsidian-style Markdown block IDs."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_BLOCK_RE = re.compile(r"(?:^|\s)\^(?P<id>\S+)")
_VALID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def summarize_unit_markdown_block_ids(units: Iterable[Any], sample_limit: int = 10) -> dict[str, Any]:
    total_units = units_with = 0
    ids: list[dict[str, str | int]] = []
    invalid: list[dict[str, str | int]] = []
    for unit in units:
        total_units += 1
        before = len(ids)
        for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            for match in _BLOCK_RE.finditer(line):
                block_id = match.group("id").strip()
                row = {"unit_id": unit_id(unit), "block_id": block_id, "line_number": line_number}
                if _VALID_RE.match(block_id):
                    ids.append(row)
                else:
                    invalid.append(row)
        if len(ids) > before:
            units_with += 1
    counts = Counter(row["block_id"] for row in ids)
    duplicate_samples = [row for row in ids if counts[row["block_id"]] > 1]
    duplicate_samples.sort(key=lambda row: (sort_key(row["block_id"]), sort_key(row["unit_id"]), int(row["line_number"])))
    invalid.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["block_id"])))
    return {
        "total_units": total_units,
        "units_with_block_ids": units_with,
        "block_id_count": len(ids),
        "duplicate_block_id_count": sum(1 for count in counts.values() if count > 1),
        "duplicate_block_id_samples": duplicate_samples[:sample_limit],
        "invalid_block_id_samples": invalid[:sample_limit],
    }
