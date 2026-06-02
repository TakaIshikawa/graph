"""Summarize Cross-Origin-Embedder-Policy headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "cross-origin-embedder-policy"
_ISOLATING = {"require-corp", "credentialless"}


def summarize_source_cross_origin_embedder_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["policy"]]
    weak = [row for row in present if row["policy"] not in _ISOLATING]
    samples = [
        {"source_id": row["source_id"], "policy": row["policy"], "field": row["field"]}
        for row in sorted(weak, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    ]
    return {
        "total_sources": len(source_list),
        "sources_with_policy": len(present),
        "policy_counts": dict(sorted(Counter(row["policy"] for row in present).items())),
        "missing_policy_count": len(source_list) - len(present),
        "isolating_policy_count": sum(1 for row in present if row["policy"] in _ISOLATING),
        "weak_policy_count": len(weak),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, str]:
    value, field = _lookup_header(source, _HEADER)
    return {"source_id": source_id(source) or str(index), "policy": field_value(value).casefold(), "field": field}


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> tuple[str, str]:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value, key
    for owner, container in (("headers", get(source, "headers")), ("response_headers", get(source, "response_headers")), ("metadata.headers", data.get("headers")), ("metadata.response_headers", data.get("response_headers"))):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value), f"{owner}.{key}"
    return "", ""
