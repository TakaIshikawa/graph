"""Summarize robots policy metadata on source records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.export._report_csv import field_value, get, metadata, sort_key, source_id

POLICY_KEYS = ("robots", "robots_policy", "robots_allowed", "crawl_allowed", "x_robots_tag", "noindex")


def summarize_source_robots_policies(sources: Iterable[Any], sample_limit: int = 5) -> dict[str, Any]:
    limit = max(0, sample_limit)
    total = allowed = disallowed = noindex = missing = 0
    policy_counts: Counter[str] = Counter()
    samples: list[dict[str, str]] = []

    for source in sources:
        total += 1
        field, raw_value = _policy_value(source)
        policy = _normalize_policy(field, raw_value)
        if policy == "missing":
            missing += 1
            continue
        policy_counts[policy] += 1
        allowed += policy == "allowed"
        disallowed += policy in {"disallowed", "noindex"}
        noindex += policy == "noindex"
        if len(samples) < limit:
            samples.append({"source_id": source_id(source), "field": field or "", "policy": policy, "value": field_value(raw_value)})

    samples.sort(key=lambda row: (sort_key(row["policy"]), sort_key(row["source_id"])))
    return {
        "total_sources": total,
        "policy_counts": {key: policy_counts[key] for key in sorted(policy_counts, key=sort_key)},
        "allowed_count": allowed,
        "disallowed_count": disallowed,
        "noindex_count": noindex,
        "missing_policy_count": missing,
        "samples": samples[:limit],
    }


def _policy_value(source: Any) -> tuple[str | None, Any]:
    meta = metadata(source)
    for key in POLICY_KEYS:
        value = meta.get(key)
        if _has_value(value):
            return f"metadata.{key}", value
    for key in POLICY_KEYS:
        value = get(source, key)
        if _has_value(value):
            return key, value
    return None, None


def _has_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    return bool(field_value(value))


def _normalize_policy(field: str | None, value: Any) -> str:
    if field is None:
        return "missing"
    key = field.rsplit(".", 1)[-1]
    if isinstance(value, bool):
        if key == "noindex":
            return "noindex" if value else "allowed"
        return "allowed" if value else "disallowed"
    text = field_value(value).casefold()
    if "noindex" in text:
        return "noindex"
    if text in {"true", "yes", "y", "1", "allow", "allowed", "all"}:
        return "allowed"
    if text in {"false", "no", "n", "0", "deny", "denied", "disallow", "disallowed", "none"}:
        return "disallowed"
    return text or "missing"
