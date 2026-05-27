"""Summarize HTML entity usage in unit content."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import get, sort_key, unit_id

_ENTITY_RE = re.compile(r"&(?:#(?P<dec>\d+)|#x(?P<hex>[0-9A-Fa-f]+)|(?P<named>[A-Za-z][A-Za-z0-9]+));")


def summarize_unit_html_entities(units: Iterable[Any], sample_limit: int = 10) -> dict[str, Any]:
    total_units = units_with = named = decimal = hex_count = 0
    frequency: Counter[str] = Counter()
    samples: list[dict[str, str | int]] = []
    for unit in units:
        total_units += 1
        found = False
        for line_number, line in enumerate(str(get(unit, "content") or "").splitlines(), start=1):
            for match in _ENTITY_RE.finditer(line):
                entity = match.group(0)
                found = True
                frequency[entity] += 1
                if match.group("named"):
                    named += 1
                elif match.group("dec"):
                    decimal += 1
                else:
                    hex_count += 1
                samples.append({"unit_id": unit_id(unit), "entity": entity, "line_number": line_number})
        if found:
            units_with += 1
    samples.sort(key=lambda row: (sort_key(row["unit_id"]), int(row["line_number"]), sort_key(row["entity"])))
    return {
        "total_units": total_units,
        "units_with_entities": units_with,
        "named_entity_count": named,
        "decimal_entity_count": decimal,
        "hex_entity_count": hex_count,
        "entity_frequency": [{"entity": entity, "count": count} for entity, count in sorted(frequency.items(), key=lambda item: (-item[1], sort_key(item[0])))],
        "samples": samples[:sample_limit],
    }
