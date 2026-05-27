"""Summarize Markdown reference-style link usages in units."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, sort_key, unit_id
from graph.export.unit_markdown_reference_usage_csv import _usages

_REF_DEF_RE = re.compile(r"^[ \t]{0,3}\[([^\]\n]+)]:")


def summarize_unit_markdown_reference_usage(units: Iterable[Any], sample_limit: int = 10) -> dict[str, Any]:
    total_units = 0
    units_with = 0
    counts = {"full": 0, "collapsed": 0, "shortcut": 0}
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total_units += 1
        content = str(get(unit, "content") or "")
        definitions = {_normalize(match.group(1)) for line in content.splitlines() if (match := _REF_DEF_RE.match(line))}
        unit_usages = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            if _REF_DEF_RE.match(line):
                continue
            for usage in _usages(line):
                unit_usages.append(usage)
                counts[usage["usage_type"]] += 1
                if _normalize(usage["label"]) not in definitions:
                    samples.append({"unit_id": unit_id(unit), "label": usage["label"], "usage_type": usage["usage_type"], "line_number": line_number})
        if unit_usages:
            units_with += 1
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), sort_key(row["label"]), int(row["line_number"])))
    return {
        "total_units": total_units,
        "units_with_reference_usages": units_with,
        "full_usage_count": counts["full"],
        "collapsed_usage_count": counts["collapsed"],
        "shortcut_usage_count": counts["shortcut"],
        "unresolved_label_samples": samples[:sample_limit],
    }


def _normalize(value: object) -> str:
    return re.sub(r"\s+", " ", field_value(value)).casefold()
