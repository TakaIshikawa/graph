"""Audit temporal ordering of evidence records."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from graph.rag._analysis_utils import result_date, result_id


def audit_evidence_temporal_ordering(evidence: Iterable[Any]) -> dict[str, Any]:
    dated = []
    undated = 0
    for index, item in enumerate(evidence or []):
        date = result_date(item)
        if date is None:
            undated += 1
        else:
            dated.append((result_id(item, index), date))

    inversions = []
    ascending = descending = True
    for (left_id, left_date), (right_id, right_date) in zip(dated, dated[1:]):
        if left_date > right_date:
            ascending = False
            inversions.append({"left_id": left_id, "left_date": left_date.isoformat(), "right_id": right_id, "right_date": right_date.isoformat()})
        if left_date < right_date:
            descending = False

    if not dated:
        ordering = "undated"
    elif ascending:
        ordering = "chronological"
    elif descending:
        ordering = "reverse_chronological"
    else:
        ordering = "mixed"

    dates = [item[1] for item in dated]
    return {
        "dated_count": len(dated),
        "undated_count": undated,
        "ordering": ordering,
        "inversions": inversions if ordering == "mixed" else [],
        "earliest_date": min(dates).isoformat() if dates else None,
        "latest_date": max(dates).isoformat() if dates else None,
    }
