"""Summarize SSL/TLS certificate expiry metadata on source records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

from graph.export._report_csv import field_value, get, metadata, parse_datetime, sort_key, source_id

EXPIRY_KEYS = ("ssl_expires_at", "tls_expires_at", "certificate_expires_at", "cert_expires_at")


def summarize_source_ssl_expiry(sources: Iterable[Any], now: Any = None, sample_limit: int = 5) -> dict[str, Any]:
    reference = _reference_datetime(now)
    soon_threshold = reference + timedelta(days=30)
    limit = max(0, sample_limit)
    total = sources_with_expiry = 0
    bucket_counts: Counter[str] = Counter({"expired": 0, "expiring_soon": 0, "valid": 0, "missing": 0, "invalid": 0})
    samples: list[dict[str, Any]] = []

    for source in sources:
        total += 1
        field, raw_value = _expiry_value(source)
        raw_text = field_value(raw_value)
        parsed = parse_datetime(raw_value)
        if not raw_text:
            bucket = "missing"
            days_until_expiry = None
        elif parsed is None:
            sources_with_expiry += 1
            bucket = "invalid"
            days_until_expiry = None
        else:
            sources_with_expiry += 1
            bucket = _bucket(parsed, reference, soon_threshold)
            days_until_expiry = (parsed - reference).days
        bucket_counts[bucket] += 1
        if len(samples) < limit and bucket != "missing":
            samples.append(
                {
                    "source_id": source_id(source),
                    "field": field or "",
                    "ssl_expires_at": raw_text,
                    "bucket": bucket,
                    "days_until_expiry": days_until_expiry,
                }
            )

    samples.sort(key=lambda row: (sort_key(row["bucket"]), sort_key(row["source_id"])))
    return {
        "total_sources": total,
        "sources_with_ssl_expiry": sources_with_expiry,
        "expired_count": bucket_counts["expired"],
        "expiring_soon_count": bucket_counts["expiring_soon"],
        "missing_ssl_expiry_count": bucket_counts["missing"],
        "bucket_counts": {key: bucket_counts[key] for key in ("expired", "expiring_soon", "valid", "missing", "invalid")},
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
