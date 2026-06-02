"""Summarize Set-Cookie prefix usage in source response headers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "set-cookie"


def summarize_source_cookie_prefixes(sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5) -> dict[str, Any]:
    total = host = secure = invalid_host = invalid_secure = unprefixed = 0
    samples: list[dict[str, str]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(sources):
        total += 1
        sid = source_id(source) or str(index)
        for cookie in _lookup_headers(source, _HEADER):
            name, attrs = _cookie(cookie)
            if not name:
                continue
            lower_attrs = {key.casefold(): value for key, value in attrs.items()}
            if name.startswith("__Host-"):
                ok = "secure" in lower_attrs and lower_attrs.get("path", "") == "/" and "domain" not in lower_attrs
                host += int(ok)
                invalid_host += int(not ok)
                kind = "host_prefix" if ok else "invalid_host_prefix"
            elif name.startswith("__Secure-"):
                ok = "secure" in lower_attrs
                secure += int(ok)
                invalid_secure += int(not ok)
                kind = "secure_prefix" if ok else "invalid_secure_prefix"
            else:
                unprefixed += 1
                kind = "unprefixed"
            if len(samples) < limit:
                samples.append({"source_id": sid, "cookie_name": name, "classification": kind})
    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["cookie_name"])))
    return {
        "total_sources": total,
        "host_prefix_count": host,
        "secure_prefix_count": secure,
        "invalid_host_prefix_count": invalid_host,
        "invalid_secure_prefix_count": invalid_secure,
        "unprefixed_cookie_count": unprefixed,
        "samples": samples[:limit],
    }


def _cookie(value: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in value.split(";") if part.strip()]
    if not parts or "=" not in parts[0]:
        return "", {}
    name = parts[0].split("=", 1)[0].strip()
    attrs: dict[str, str] = {}
    for part in parts[1:]:
        key, _, raw = part.partition("=")
        attrs[key.strip()] = raw.strip()
    return name, attrs


def _lookup_headers(source: Mapping[str, Any] | object, header: str) -> list[str]:
    values: list[str] = []
    data = metadata(source)
    for container in (source, data, get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if not isinstance(container, Mapping):
            continue
        for key, value in container.items():
            if str(key).casefold().replace("_", "-") == header:
                _append(values, value)
    return values


def _append(values: list[str], raw: Any) -> None:
    if isinstance(raw, list | tuple | set):
        for item in raw:
            _append(values, item)
        return
    value = field_value(raw)
    if value:
        values.append(value)
