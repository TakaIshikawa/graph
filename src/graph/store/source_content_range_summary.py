"""Summarize Content-Range headers in sources."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "content-range"
_CONTENT_RANGE_RE = re.compile(r"^\s*(?P<unit>[A-Za-z][A-Za-z0-9._-]*)\s+(?P<range>\*|(?P<start>\d+)-(?P<end>\d+))/(?:\s*)(?P<total>\d+|\*)\s*$")


def summarize_source_content_ranges(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Return compact counts and samples for HTTP Content-Range values."""
    source_list = list(sources)
    limit = max(0, sample_limit)
    rows: list[dict[str, Any]] = []
    unit_counts: Counter[str] = Counter()
    sources_with = unsatisfied_count = unknown_total_count = complete_count = malformed_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source)
        if not value:
            continue
        sources_with += 1
        parsed = _parse_content_range(value)
        unit_counts[parsed["unit"]] += 1
        unsatisfied_count += int(parsed["unsatisfied"])
        unknown_total_count += int(parsed["unknown_total"])
        complete_count += int(parsed["complete"])
        malformed_count += int(parsed["malformed"])
        sample = {"source_id": sid, "raw": value, "unit": parsed["unit"]}
        if parsed["malformed"]:
            sample["malformed"] = True
        elif parsed["unsatisfied"]:
            sample.update({"unsatisfied": True, "total": parsed["total"]})
        else:
            sample.update({"start": parsed["start"], "end": parsed["end"], "total": parsed["total"]})
        rows.append(sample)

    samples = sorted(rows, key=lambda row: sort_key(row["source_id"]))[:limit]
    return {
        "total_sources": len(source_list),
        "sources_with_content_range": sources_with,
        "missing_content_range_count": len(source_list) - sources_with,
        "unit_counts": {key: unit_counts[key] for key in sorted(unit_counts, key=sort_key)},
        "complete_count": complete_count,
        "unsatisfied_count": unsatisfied_count,
        "unknown_total_count": unknown_total_count,
        "malformed_count": malformed_count,
        "samples": samples,
    }


def _parse_content_range(value: str) -> dict[str, Any]:
    match = _CONTENT_RANGE_RE.match(value)
    if not match:
        return {
            "unit": "unknown",
            "start": None,
            "end": None,
            "total": None,
            "unsatisfied": False,
            "unknown_total": True,
            "complete": False,
            "malformed": True,
        }
    unit = match.group("unit").casefold()
    total_text = match.group("total")
    total = None if total_text == "*" else int(total_text)
    unknown_total = total is None
    if match.group("range") == "*":
        return {
            "unit": unit,
            "start": None,
            "end": None,
            "total": total,
            "unsatisfied": True,
            "unknown_total": unknown_total,
            "complete": False,
            "malformed": False,
        }
    start = int(match.group("start"))
    end = int(match.group("end"))
    malformed = end < start
    return {
        "unit": unit if not malformed else "unknown",
        "start": start,
        "end": end,
        "total": total,
        "unsatisfied": False,
        "unknown_total": unknown_total,
        "complete": bool(total is not None and start == 0 and end + 1 == total and not malformed),
        "malformed": malformed,
    }


def _lookup_header(source: Mapping[str, Any] | object) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (_HEADER, _HEADER.replace("-", "_"), _HEADER.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == _HEADER:
                    return field_value(value)
    return ""
