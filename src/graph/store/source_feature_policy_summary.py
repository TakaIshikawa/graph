"""Summarize legacy Feature-Policy headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "feature-policy"


def summarize_source_feature_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    feature_counts: Counter[str] = Counter()
    disabled_feature_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []
    sources_with = malformed_directive_count = 0

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        sources_with += 1
        for directive in _split_quoted(value, ";"):
            raw = directive.strip()
            if not raw:
                continue
            parts = raw.split(None, 1)
            feature = field_value(parts[0]).casefold()
            if not feature or len(parts) < 2:
                malformed_directive_count += 1
                continue
            allowlist = field_value(parts[1])
            feature_counts[feature] += 1
            if _disabled(allowlist):
                disabled_feature_counts[feature] += 1
            if len(samples) < limit:
                samples.append({"source_id": sid, "feature": feature, "allowlist": allowlist})

    return {
        "total_sources": len(source_list),
        "sources_with_feature_policy": sources_with,
        "feature_counts": {key: feature_counts[key] for key in sorted(feature_counts, key=sort_key)},
        "disabled_feature_counts": {key: disabled_feature_counts[key] for key in sorted(disabled_feature_counts, key=sort_key)},
        "malformed_directive_count": malformed_directive_count,
        "missing_feature_policy_count": len(source_list) - sources_with,
        "samples": samples,
    }


def _disabled(allowlist: str) -> bool:
    text = allowlist.strip()
    return not text or text.casefold() in {"'none'", "none"}


def _split_quoted(value: str, delimiter: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote = False
    for char in value:
        if char == '"':
            quote = not quote
        if char == delimiter and not quote:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(char)
    parts.append("".join(buf))
    return parts


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
