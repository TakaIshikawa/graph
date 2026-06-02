"""Summarize HTTP Priority response/request headers on sources."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "priority"
_ALIASES = {_HEADER, "Priority", "http_priority", "priority_header"}
_CONTAINERS = ("headers", "response_headers", "request_headers")


def summarize_source_priority_headers(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    """Summarize HTTP Priority header urgency and incremental hints."""
    limit = max(0, sample_limit)
    total = missing = malformed = incremental_true = 0
    urgency_counts: Counter[str] = Counter()
    samples: list[dict[str, str | bool | None]] = []

    for index, source in enumerate(sources):
        total += 1
        field, value = _priority_value(source)
        header = field_value(value)
        sid = source_id(source) or str(index)
        if not header:
            missing += 1
            continue

        parsed = _parse_priority(header)
        if parsed["malformed"]:
            malformed += 1
        urgency = parsed["urgency"]
        if urgency is not None:
            urgency_counts[urgency] += 1
        if parsed["incremental"]:
            incremental_true += 1
        if len(samples) < limit:
            samples.append(
                {
                    "source_id": sid,
                    "field": field or "",
                    "priority": header,
                    "urgency": urgency,
                    "incremental": parsed["incremental"],
                    "malformed": parsed["malformed"],
                }
            )

    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["priority"])))
    return {
        "total_sources": total,
        "urgency_counts": {key: urgency_counts[key] for key in sorted(urgency_counts, key=sort_key)},
        "incremental_true_count": incremental_true,
        "malformed_count": malformed,
        "missing_header_count": missing,
        "samples": samples[:limit],
    }


def _priority_value(source: Mapping[str, Any] | object) -> tuple[str | None, Any]:
    meta = metadata(source)
    for key in _ALIASES:
        value = get(source, key)
        if field_value(value):
            return key, value
    for key in _ALIASES:
        value = meta.get(key)
        if field_value(value):
            return f"metadata.{key}", value
    for container_name in _CONTAINERS:
        container = get(source, container_name)
        found = _container_header(container)
        if found:
            return f"{container_name}.priority", found
        found = _container_header(meta.get(container_name))
        if found:
            return f"metadata.{container_name}.priority", found
    return None, None


def _container_header(container: object) -> object:
    if not isinstance(container, Mapping):
        return None
    for key, value in container.items():
        if str(key).casefold().replace("_", "-") == _HEADER:
            return value
    return None


def _parse_priority(value: str) -> dict[str, str | bool | None]:
    urgency: str | None = None
    incremental = False
    malformed = False
    seen_names: set[str] = set()

    for part in value.split(","):
        item = part.strip()
        if not item:
            malformed = True
            continue
        if "=" in item:
            name, raw_param = item.split("=", 1)
            name = name.strip().casefold()
            param = raw_param.strip()
            if not name or not param:
                malformed = True
                continue
        else:
            name = item.casefold()
            param = None

        if name in seen_names:
            malformed = True
        seen_names.add(name)

        if name == "u":
            if param is not None and param.isdigit() and 0 <= int(param) <= 7:
                urgency = str(int(param))
            else:
                malformed = True
        elif name == "i":
            if param is None:
                incremental = True
            elif param.casefold() in {"true", "1"}:
                incremental = True
            elif param.casefold() in {"false", "0"}:
                incremental = False
            else:
                malformed = True
        else:
            malformed = True

    return {"urgency": urgency, "incremental": incremental, "malformed": malformed}
