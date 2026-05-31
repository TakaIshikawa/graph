"""Summarize authentication expiry dates on source records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, sort_key, source_id

EXPIRY_KEYS = ("token_expires_at", "expires_at", "credential_expires_at", "oauth_expires_at", "auth_expires_at")


def summarize_source_auth_expiry(sources: Iterable[Any], reference_date: Any = None, sample_limit: int = 5) -> dict[str, Any]:
    reference = _reference_datetime(reference_date)
    soon_threshold = reference + timedelta(days=14)
    limit = max(0, sample_limit)
    total = 0
    buckets: Counter[str] = Counter({"expired": 0, "expiring_soon": 0, "valid": 0, "unknown": 0})
    field_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for source in sources:
        total += 1
        field, raw_value = _expiry_value(source)
        expires_at = parse_datetime(raw_value)
        if expires_at is None:
            bucket = "unknown"
            reason = "missing" if field is None else "invalid"
            days_until_expiry = None
        else:
            bucket = _bucket(expires_at, reference, soon_threshold)
            reason = ""
            days_until_expiry = (expires_at - reference).days
        buckets[bucket] += 1
        if field:
            field_counts[field] += 1
        if len(samples) < limit:
            samples.append(
                {
                    "source_id": source_id(source),
                    "field": field or "",
                    "expires_at": field_value(raw_value),
                    "bucket": bucket,
                    "reason": reason,
                    "days_until_expiry": days_until_expiry,
                }
            )

    samples.sort(key=lambda row: (sort_key(row["bucket"]), sort_key(row["source_id"])))
    return {
        "total_sources": total,
        "bucket_counts": {key: buckets[key] for key in ("expired", "expiring_soon", "valid", "unknown")},
        "field_counts": {key: field_counts[key] for key in sorted(field_counts, key=sort_key)},
        "reference_date": reference.isoformat(),
        "expiring_soon_days": 14,
        "samples": samples[:limit],
    }


def _reference_datetime(value: Any) -> datetime:
    parsed = parse_datetime(value)
    return parsed if parsed is not None else datetime.now(timezone.utc)


def _expiry_value(source: Any) -> tuple[str | None, Any]:
    meta = metadata(source)
    for key in EXPIRY_KEYS:
        value = meta.get(key)
        if field_value(value):
            return f"metadata.{key}", value
    for key in EXPIRY_KEYS:
        value = get(source, key)
        if field_value(value):
            return key, value
    return None, None


def _bucket(expires_at: datetime, reference: datetime, soon_threshold: datetime) -> str:
    if expires_at < reference:
        return "expired"
    if expires_at <= soon_threshold:
        return "expiring_soon"
    return "valid"
