"""Summarize Permissions-Policy headers in sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "permissions-policy"
_RISKY = {"camera", "microphone", "geolocation", "payment"}


def summarize_source_permissions_policies(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows = [_row(source, index) for index, source in enumerate(source_list)]
    present = [row for row in rows if row["policy"]]
    samples = sorted(present, key=lambda row: sort_key(row["source_id"]))[: max(0, sample_limit)]
    return {
        "total_sources": len(source_list),
        "sources_with_policy": len(present),
        "missing_policy_count": len(source_list) - len(present),
        "directive_counts": dict(sorted(Counter(directive for row in present for directive in row["directives"]).items())),
        "risky_allowance_count": sum(row["risky_allowance_count"] for row in present),
        "empty_policy_count": sum(1 for row in present if not row["directives"]),
        "invalid_fragment_count": sum(row["invalid_fragment_count"] for row in present),
        "rows": sorted(present, key=lambda row: sort_key(row["source_id"])),
        "samples": samples,
    }


def _row(source: Mapping[str, Any] | object, index: int) -> dict[str, Any]:
    policy = field_value(_lookup_header(source, _HEADER))
    directives, invalid, risky = _parse_policy(policy)
    return {
        "source_id": source_id(source) or str(index),
        "policy": policy,
        "directives": directives,
        "invalid_fragment_count": invalid,
        "risky_allowance_count": risky,
    }


def _parse_policy(policy: str) -> tuple[list[str], int, int]:
    directives: list[str] = []
    invalid = 0
    risky = 0
    for fragment in [part.strip() for part in policy.split(",") if part.strip()]:
        name, _, allowlist = fragment.partition("=")
        name = name.strip().casefold()
        if not name or not allowlist.startswith("(") or not allowlist.endswith(")"):
            invalid += 1
            continue
        directives.append(name)
        normalized_allowlist = allowlist.casefold()
        if name in _RISKY and ("*" in normalized_allowlist or "self" in normalized_allowlist):
            risky += 1
    return directives, invalid, risky


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> Any:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title(), "Permissions-Policy"):
            value = get(container, key) if container_name == "source" else container.get(key)
            if field_value(value):
                return value
    for container in (get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return value
    return ""
