"""Summarize JSON-LD structured data in sources."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_SCRIPT_RE = re.compile(r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.IGNORECASE | re.DOTALL)
_TYPE_RE = re.compile(r"""\stype\s*=\s*(?:"application/ld\+json"|'application/ld\+json'|application/ld\+json)""", re.IGNORECASE)


def summarize_source_json_ld(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    type_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = invalid = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        payloads = _payloads(source)
        source_has_valid = False
        for payload in payloads:
            try:
                parsed = json.loads(payload) if isinstance(payload, str) else payload
            except (TypeError, ValueError):
                invalid += 1
                continue
            types = sorted(_types(parsed), key=sort_key)
            if not types:
                continue
            source_has_valid = True
            type_counts.update(types)
            for item_type in types:
                if len(samples) < limit:
                    samples.append({"source_id": sid, "type": item_type})
        if source_has_valid:
            sources_with += 1

    return {
        "total_sources": len(source_list),
        "sources_with_json_ld": sources_with,
        "missing_json_ld_count": len(source_list) - sources_with,
        "invalid_json_ld_count": invalid,
        "type_counts": {key: type_counts[key] for key in sorted(type_counts, key=sort_key)},
        "samples": samples,
    }


def _payloads(source: Mapping[str, Any] | object) -> list[object]:
    data = metadata(source)
    payloads: list[object] = []
    for key in ("json_ld", "jsonld", "structured_data"):
        value = get(source, key) or data.get(key)
        if isinstance(value, list):
            payloads.extend(value)
        elif value:
            payloads.append(value)
    html = field_value(get(source, "html") or data.get("html") or data.get("content"))
    payloads.extend(match.group("body").strip() for match in _SCRIPT_RE.finditer(html) if _TYPE_RE.search(match.group("attrs")))
    return payloads


def _types(value: object) -> list[str]:
    if isinstance(value, Mapping):
        raw_type = value.get("@type")
        found = [field_value(item) for item in (raw_type if isinstance(raw_type, list) else [raw_type]) if field_value(item)]
        for key in ("@graph", "graph"):
            found.extend(_types(value.get(key)))
        return found
    if isinstance(value, list):
        return [item_type for item in value for item_type in _types(item)]
    return []
