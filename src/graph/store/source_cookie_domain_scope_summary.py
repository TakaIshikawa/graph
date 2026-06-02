"""Summarize Set-Cookie Domain scope usage in source response headers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id
from graph.store.source_cookie_prefix_summary import _cookie

_COMMON_SUFFIXES = {"com", "org", "net", "edu", "gov", "co.uk", "ac.uk"}


def summarize_source_cookie_domain_scopes(
    sources: Iterable[Mapping[str, Any] | object], sample_limit: int = 5
) -> dict[str, Any]:
    total = host_only = domain = public_suffix_like = 0
    samples: list[dict[str, str]] = []
    limit = max(0, sample_limit)
    for index, source in enumerate(sources):
        total += 1
        sid = source_id(source) or str(index)
        for cookie in _lookup_headers(source, "set-cookie"):
            name, attrs = _cookie(cookie)
            if not name:
                continue
            attr_map = {key.casefold(): value for key, value in attrs.items()}
            raw_domain = attr_map.get("domain", "").strip()
            if raw_domain:
                domain += 1
                normalized = raw_domain.lstrip(".").casefold()
                broad = _public_suffix_like(normalized)
                public_suffix_like += int(broad)
                classification = "public_suffix_like" if broad else "domain"
            else:
                host_only += 1
                normalized = ""
                classification = "host_only"
            if len(samples) < limit:
                samples.append({"source_id": sid, "cookie_name": name, "domain": normalized, "classification": classification})
    samples.sort(key=lambda row: (sort_key(row["source_id"]), sort_key(row["cookie_name"])))
    return {
        "total_sources": total,
        "host_only_cookie_count": host_only,
        "domain_cookie_count": domain,
        "public_suffix_like_domain_count": public_suffix_like,
        "samples": samples[:limit],
    }


def _public_suffix_like(domain: str) -> bool:
    return domain in _COMMON_SUFFIXES or domain.count(".") == 0


def _lookup_headers(source: Mapping[str, Any] | object, header: str) -> list[str]:
    values: list[str] = []
    data = metadata(source)
    for container in (source, data, get(source, "headers"), get(source, "response_headers"), data.get("headers"), data.get("response_headers")):
        if isinstance(container, Mapping):
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
