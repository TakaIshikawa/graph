"""Summarize Content-Security-Policy-Report-Only values on source records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

_HEADER = "content-security-policy-report-only"
_UNSAFE_TOKENS = ("'unsafe-inline'", "'unsafe-eval'")


def summarize_source_content_security_policy_report_only(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    source_list = list(sources)
    rows: list[dict[str, Any]] = []
    directive_counts: Counter[str] = Counter()
    report_uri_counts: Counter[str] = Counter()
    report_to_counts: Counter[str] = Counter()
    unsafe_directive_counts: Counter[str] = Counter()
    unsafe_token_counts: Counter[str] = Counter()
    for index, source in enumerate(source_list):
        sid = source_id(source) or str(index)
        for policy in _policy_values(source):
            directives = _directives(policy)
            row = {"source_id": sid, "value": policy, "directive_count": len(directives), "directives": [name for name, _ in directives]}
            rows.append(row)
            for name, values in directives:
                directive_counts[name] += 1
                if name == "report-uri":
                    for value in values:
                        report_uri_counts[value] += 1
                if name == "report-to":
                    for value in values:
                        report_to_counts[value] += 1
                for token in _UNSAFE_TOKENS:
                    if token in values:
                        unsafe_directive_counts[name] += 1
                        unsafe_token_counts[token.strip("'")] += 1
    rows.sort(key=lambda row: sort_key(row["source_id"]))
    limit = max(0, sample_limit)
    return {
        "total_sources": len(source_list),
        "sources_with_content_security_policy_report_only": len({row["source_id"] for row in rows}),
        "missing_content_security_policy_report_only_count": len(source_list) - len({row["source_id"] for row in rows}),
        "directive_counts": dict(sorted(directive_counts.items(), key=lambda item: sort_key(item[0]))),
        "report_uri_counts": dict(sorted(report_uri_counts.items(), key=lambda item: sort_key(item[0]))),
        "report_to_counts": dict(sorted(report_to_counts.items(), key=lambda item: sort_key(item[0]))),
        "unsafe_directive_counts": dict(sorted(unsafe_directive_counts.items(), key=lambda item: sort_key(item[0]))),
        "unsafe_token_counts": dict(sorted(unsafe_token_counts.items(), key=lambda item: sort_key(item[0]))),
        "rows": rows,
        "samples": rows[:limit],
    }


def _policy_values(source: Any) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for scope in (source, metadata(source)):
        for key in (_HEADER, _HEADER.replace("-", "_"), _HEADER.title()):
            _append(values, seen, get(scope, key) if scope is source else scope.get(key))
        for container in _header_maps(scope):
            for key, value in container.items():
                if field_value(key).casefold().replace("_", "-") == _HEADER:
                    _append(values, seen, value)
    return values


def _header_maps(value: Any) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("headers", "response_headers", "http_headers", "metadata_headers"):
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


def _directives(policy: str) -> list[tuple[str, list[str]]]:
    directives: list[tuple[str, list[str]]] = []
    for part in policy.split(";"):
        tokens = [token.strip().casefold() for token in part.split() if token.strip()]
        if tokens:
            directives.append((tokens[0], tokens[1:]))
    return directives
