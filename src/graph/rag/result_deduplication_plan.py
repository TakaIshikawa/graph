"""Build a deterministic deduplication plan for RAG results."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse, urlunparse

from graph.rag._analysis_utils import content_text, result_id, string, value


def build_result_deduplication_plan(results: Iterable[Any]) -> dict[str, Any]:
    """Identify duplicate URL/path or title/content fingerprints."""
    groups: dict[str, list[tuple[str, int]]] = defaultdict(list)
    seen = list(results or [])
    for index, result in enumerate(seen):
        fingerprint = _fingerprint(result)
        groups[fingerprint].append((result_id(result, index), index))

    duplicate_groups = []
    canonical_ids = []
    dropped_ids = []
    reasons = {}
    for fingerprint, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        ordered = sorted(ids, key=lambda item: (item[0], item[1]))
        canonical = ordered[0][0]
        dropped = [item[0] for item in ordered[1:]]
        canonical_ids.append(canonical)
        dropped_ids.extend(dropped)
        reason = "matching_url_or_path" if fingerprint.startswith(("url:", "path:")) else "matching_title_content"
        reasons[canonical] = reason
        duplicate_groups.append({"fingerprint": fingerprint, "canonical_result_id": canonical, "duplicate_result_ids": dropped, "result_count": len(ids)})

    return {
        "duplicate_groups": duplicate_groups,
        "canonical_result_ids": canonical_ids,
        "dropped_result_ids": dropped_ids,
        "duplicate_count": len(dropped_ids),
        "retention_reasons": reasons,
    }


def _fingerprint(result: Any) -> str:
    for key in ("url", "source_url", "canonical_url"):
        text = string(value(result, key))
        if text:
            return f"url:{_normalize_url(text)}"
    for key in ("source_path", "path"):
        text = string(value(result, key))
        if text:
            return f"path:{text.strip().rstrip('/').casefold()}"
    title = _normalize_text(value(result, "title"))
    body = _normalize_text(content_text(result))
    return f"text:{title}|{body}"


def _normalize_url(url: str) -> str:
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.netloc.casefold()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    return urlunparse((parsed.scheme.casefold() or "https", host, path, "", "", ""))


def _normalize_text(text: Any) -> str:
    return re.sub(r"\W+", " ", string(text) or "").strip().casefold()
