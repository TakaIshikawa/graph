"""Analyze provenance completeness for evidence records.

Default required fields are source, url, title, author, published_at,
retrieved_at, and source_type.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import has_evidence, result_id, value

DEFAULT_REQUIRED_FIELDS = ("source", "url", "title", "author", "published_at", "retrieved_at", "source_type")


def analyze_evidence_provenance_completeness(
    evidence: Iterable[Any] | None = None,
    required_fields: Iterable[str] | None = None,
    sample_limit: int = 5,
) -> dict[str, Any]:
    """Return aggregate provenance completeness and missing-field samples."""
    fields = _required_fields(required_fields)
    rows = list(evidence or [])
    limit = max(0, int(sample_limit))
    missing_counts: Counter[str] = Counter({field: 0 for field in fields})
    complete_count = 0
    completeness_sum = 0.0
    samples: list[dict[str, Any]] = []

    for index, record in enumerate(rows):
        missing = [field for field in fields if not has_evidence(value(record, field))]
        present_count = len(fields) - len(missing)
        completeness_sum += present_count / len(fields) if fields else 1.0
        if missing:
            for field in missing:
                missing_counts[field] += 1
            if len(samples) < limit:
                samples.append({"result_id": result_id(record, index), "missing_fields": missing})
        else:
            complete_count += 1

    record_count = len(rows)
    return {
        "record_count": record_count,
        "required_fields": list(fields),
        "complete_record_count": complete_count,
        "average_completeness": round(completeness_sum / record_count, 3) if record_count else 0.0,
        "missing_field_counts": dict(missing_counts),
        "samples": samples,
    }


def _required_fields(required_fields: Iterable[str] | None) -> tuple[str, ...]:
    if required_fields is None:
        return DEFAULT_REQUIRED_FIELDS
    fields: list[str] = []
    for field in required_fields:
        normalized = str(field).strip()
        if normalized and normalized not in fields:
            fields.append(normalized)
    return tuple(fields)
