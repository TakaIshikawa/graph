"""Analyze canonical source duplication in RAG result sets."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from graph.rag._analysis_utils import result_id, string, value

_URL_KEYS = (
    "source_url",
    "url",
    "canonical_url",
    "external_url",
    "link",
    "permalink",
    "uri",
)
_DOMAIN_KEYS = ("domain", "source_domain")
_TRACKING_PARAMS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "mkt_tok",
    "ref",
    "ref_src",
    "spm",
    "utm_campaign",
    "utm_content",
    "utm_medium",
    "utm_source",
    "utm_term",
    "utm_id",
}


def analyze_result_source_canonicalization(results: Iterable[Any]) -> dict[str, Any]:
    """Group results that resolve to the same canonical source URL or domain."""
    canonical_rows: list[dict[str, Any]] = []
    for index, result in enumerate(results or []):
        raw_source = _source_value(result)
        canonical = _canonical_source(raw_source) if raw_source else None
        if canonical is None:
            continue
        canonical_rows.append(
            {
                "result_id": result_id(result, index),
                "canonical_source": canonical,
                "source": raw_source,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        grouped[row["canonical_source"]].append(row)

    canonical_groups = [
        {
            "canonical_source": canonical,
            "result_ids": [row["result_id"] for row in rows],
            "source_values": sorted({row["source"] for row in rows}),
        }
        for canonical, rows in sorted(grouped.items())
        if len(rows) > 1
    ]
    duplicate_result_ids = [
        result_id
        for group in canonical_groups
        for result_id in group["result_ids"][1:]
    ]
    return {
        "canonical_groups": canonical_groups,
        "duplicate_result_ids": duplicate_result_ids,
        "unique_source_count": len(grouped),
        "recommendations": _recommendations(canonical_groups),
    }


def _source_value(result: Any) -> str | None:
    for key in _URL_KEYS:
        text = string(value(result, key))
        if text:
            return text
    for key in _DOMAIN_KEYS:
        text = string(value(result, key))
        if text:
            return text
    return None


def _canonical_source(raw_source: str | None) -> str | None:
    text = string(raw_source)
    if text is None:
        return None
    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None

    path = parsed.path or ""
    if path != "/":
        path = path.rstrip("/")
    else:
        path = ""

    query_pairs = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not _is_tracking_param(key)
    ]
    query = urlencode(sorted(query_pairs), doseq=True)
    return urlunparse((parsed.scheme.casefold() or "https", host, path, "", query, ""))


def _is_tracking_param(key: str) -> bool:
    lowered = key.casefold()
    return lowered in _TRACKING_PARAMS or lowered.startswith("utm_")


def _recommendations(canonical_groups: list[dict[str, Any]]) -> list[str]:
    if not canonical_groups:
        return []
    return [
        "collapse_duplicate_canonical_sources_before_answering",
        "replace_duplicate_results_with_distinct_sources_when_source_diversity_matters",
    ]
