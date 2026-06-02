"""Summarize legacy X-XSS-Protection headers in sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "x-xss-protection"


def summarize_source_x_xss_protections(
    sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5
) -> dict[str, Any]:
    source_list = list(sources)
    limit = max(0, sample_limit)
    enabled = disabled = block_mode = report_uri = invalid = 0
    samples: list[dict[str, str]] = []

    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        value = _lookup_header(source, _HEADER)
        if not value:
            continue
        parsed = _parse(value)
        if parsed["valid"] == "false":
            invalid += 1
        elif parsed["enabled"] == "true":
            enabled += 1
            if parsed["block_mode"] == "true":
                block_mode += 1
            if parsed["report_uri"]:
                report_uri += 1
        else:
            disabled += 1
        if len(samples) < limit:
            samples.append({"source_id": sid, "value": value, **parsed})

    samples.sort(key=lambda row: sort_key(row["source_id"]))
    present = enabled + disabled + invalid
    return {
        "total_sources": len(source_list),
        "sources_with_x_xss_protection": present,
        "enabled_count": enabled,
        "disabled_count": disabled,
        "block_mode_count": block_mode,
        "report_uri_count": report_uri,
        "invalid_value_count": invalid,
        "missing_x_xss_protection_count": len(source_list) - present,
        "samples": samples[:limit],
    }


def _parse(value: str) -> dict[str, str]:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    first = parts[0].casefold() if parts else ""
    if first == "0" and len(parts) == 1:
        return {"enabled": "false", "block_mode": "false", "report_uri": "", "valid": "true"}
    if first != "1":
        return {"enabled": "false", "block_mode": "false", "report_uri": "", "valid": "false"}
    block = False
    report = ""
    valid = True
    for part in parts[1:]:
        key, _, raw = part.partition("=")
        name = key.strip().casefold()
        if name == "mode" and raw.strip().casefold() == "block":
            block = True
        elif name == "report" and raw.strip():
            report = raw.strip()
        else:
            valid = False
    return {"enabled": "true", "block_mode": str(block).lower(), "report_uri": report, "valid": str(valid).lower()}


def _lookup_header(source: Mapping[str, Any] | object, header: str) -> str:
    data = metadata(source)
    for container_name, container in (("source", source), ("metadata", data)):
        for key in (header, header.replace("-", "_"), header.title()):
            value = field_value(get(container, key) if container_name == "source" else container.get(key))
            if value:
                return value
    for container in (
        get(source, "headers"),
        get(source, "response_headers"),
        data.get("headers"),
        data.get("response_headers"),
    ):
        if isinstance(container, Mapping):
            for key, value in container.items():
                if str(key).casefold().replace("_", "-") == header:
                    return field_value(value)
    return ""
