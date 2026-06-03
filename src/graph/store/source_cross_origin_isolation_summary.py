"""Summarize cross-origin isolation readiness from COOP, COEP, and CORP headers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADERS = {
    "coop": "cross-origin-opener-policy",
    "coep": "cross-origin-embedder-policy",
    "corp": "cross-origin-resource-policy",
}
_COOP_ISOLATING = {"same-origin"}
_COEP_ISOLATING = {"require-corp", "credentialless"}


def summarize_source_cross_origin_isolation(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    rows.sort(key=lambda row: sort_key(row["source_id"]))
    candidates = [row for row in rows if row["isolated_candidate"]]
    return {
        "total_sources": len(source_list),
        "isolated_candidate_count": len(candidates),
        "missing_policy_counts": {name: sum(1 for row in rows if not row[name]) for name in ("coop", "coep", "corp")},
        "rows": rows,
        "samples": rows[: max(0, sample_limit)],
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    values = {name: _lookup_header(source, header).casefold() for name, header in _HEADERS.items()}
    return {
        "source_id": source_id(source) or str(index),
        "coop": values["coop"],
        "coep": values["coep"],
        "corp": values["corp"],
        "isolated_candidate": values["coop"] in _COOP_ISOLATING and values["coep"] in _COEP_ISOLATING,
    }


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
