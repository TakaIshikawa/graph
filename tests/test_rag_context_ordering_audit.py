from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from graph.rag.context_ordering_audit import audit_context_ordering


@dataclass
class Result:
    id: str
    score: float
    metadata: dict


def test_context_ordering_audit_reports_score_date_and_source_risks():
    report = audit_context_ordering(
        [
            {"id": "old-low", "score": 0.4, "date": "2024-01-01", "url": "https://same.test/a"},
            {"id": "fresh-high", "score": 0.8, "date": "2025-01-01", "url": "https://same.test/b"},
            {"id": "third", "score": 0.7, "date": "2025-02-01", "url": "https://same.test/c"},
        ],
        now=date(2026, 1, 1),
    )

    assert [issue["type"] for issue in report["issues"]] == [
        "low-score-before-high-score",
        "stale-before-fresh",
        "same-source-top-block",
    ]
    assert report["summary"] == {"result_count": 3, "issue_count": 3}


def test_context_ordering_audit_accepts_objects_tuples_and_metadata_fields():
    report = audit_context_ordering(
        [
            (Result("a", 0.3, {"published_at": "2023-01-01", "source": "docs"}), 0.3),
            Result("b", 0.7, {"published_at": "2024-01-01", "source": "blog"}),
        ]
    )

    assert report["issues"] == [
        {"type": "low-score-before-high-score", "severity": "medium", "result_ids": ["a", "b"]},
        {"type": "stale-before-fresh", "severity": "medium", "result_ids": ["a", "b"]},
    ]


def test_context_ordering_audit_validates_now():
    with pytest.raises(ValueError, match="now must be a date, datetime, or None"):
        audit_context_ordering([], now="2026-01-01")
