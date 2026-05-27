"""Audit dataset freshness for evidence records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import coerce_now, parse_date, result_id, value

_DATE_KEYS = ("dataset_date", "data_through", "collected_at", "updated_at", "published_at")


def audit_evidence_dataset_freshness(evidence: Iterable[Any], reference_date: Any = None) -> dict[str, Any]:
    ref = coerce_now(reference_date)
    buckets = {"fresh": [], "aging": [], "stale": [], "unknown": []}
    records = []
    for index, item in enumerate(evidence):
        field, parsed = _record_date(item)
        bucket = "unknown"
        age_days = None
        if parsed is not None:
            age_days = (ref - parsed).days
            bucket = "fresh" if age_days <= 180 else "aging" if age_days <= 365 else "stale"
        rid = result_id(item, index)
        buckets[bucket].append(rid)
        records.append({"id": rid, "date_field": field, "date": parsed.isoformat() if parsed else None, "age_days": age_days, "bucket": bucket})
    return {"total_evidence": len(records), "records": records, "bucket_counts": {k: len(v) for k, v in buckets.items()}, "buckets": buckets}


def _record_date(item: Any) -> tuple[str | None, Any]:
    for key in _DATE_KEYS:
        parsed = parse_date(value(item, key))
        if parsed is not None:
            return key, parsed
    return None, None
