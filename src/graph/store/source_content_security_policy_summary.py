"""Summarize Content-Security-Policy values on source records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_CSP_FIELD_KEYS = (
    "content_security_policy",
    "content-security-policy",
    "csp",
)
_HEADER_KEYS = ("headers", "response_headers", "http_headers", "metadata_headers")
_HEADER_CSP_KEYS = ("content-security-policy", "content-security-policy-report-only")
_UNSAFE_TOKENS = ("'unsafe-inline'", "'unsafe-eval'")


def summarize_source_content_security_policies(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = with_policies = 0
    directive_counts: Counter[str] = Counter()
    unsafe_directive_counts: Counter[str] = Counter()
    unsafe_token_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for source in sources:
        total += 1
        policies = _policy_values(source)
        if not policies:
            continue
        with_policies += 1
        for policy in policies:
            directives = _directives(policy)
            for name, values in directives:
                directive_counts[name] += 1
                for token in _UNSAFE_TOKENS:
                    if token in values:
                        unsafe_directive_counts[name] += 1
                        unsafe_token_counts[token.strip("'")] += 1
            if len(samples) < limit:
                samples.append(
                    {
                        "source_id": source_id(source),
                        "directive_count": len(directives),
                        "directives": [name for name, _ in directives],
                        "has_unsafe_inline": any("'unsafe-inline'" in values for _, values in directives),
                        "has_unsafe_eval": any("'unsafe-eval'" in values for _, values in directives),
                    }
                )

    samples.sort(key=lambda row: sort_key(row["source_id"]))
    return {
        "total_sources": total,
        "sources_with_content_security_policy": with_policies,
        "directive_counts": {key: directive_counts[key] for key in sorted(directive_counts, key=sort_key)},
        "unsafe_directive_counts": {key: unsafe_directive_counts[key] for key in sorted(unsafe_directive_counts, key=sort_key)},
        "unsafe_token_counts": {key: unsafe_token_counts[key] for key in sorted(unsafe_token_counts, key=sort_key)},
        "samples": samples[:limit],
    }


def _policy_values(source: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for scope in (source, metadata(source)):
        for key in _CSP_FIELD_KEYS:
            _append(values, seen, get(scope, key) if scope is source else scope.get(key))
        for header_map in _header_maps(scope):
            for key, value in header_map.items():
                if field_value(key).casefold() in _HEADER_CSP_KEYS:
                    _append(values, seen, value)
    return values


def _header_maps(value: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in _HEADER_KEYS:
        raw = get(value, key) if not isinstance(value, Mapping) else value.get(key)
        if isinstance(raw, Mapping):
            rows.append(raw)
    return rows


def _append(values: list[str], seen: set[str], raw: Any) -> None:
    if isinstance(raw, list | tuple | set):
        for item in raw:
            _append(values, seen, item)
        return
    text = field_value(raw)
    if text and text not in seen:
        values.append(text)
        seen.add(text)


def _directives(policy: str) -> list[tuple[str, set[str]]]:
    directives: list[tuple[str, set[str]]] = []
    for part in policy.split(";"):
        tokens = [token.strip() for token in part.split() if token.strip()]
        if not tokens:
            continue
        directives.append((tokens[0].casefold(), {token.casefold() for token in tokens[1:]}))
    return directives
