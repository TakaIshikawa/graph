from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from graph.rag.context_risk_flags import flag_context_risks


def test_flag_context_risks_returns_ordered_risks_and_summary():
    report = flag_context_risks(
        [
            {"id": "a", "url": "https://same.test/a", "content": "Short", "date": "2024-01-01"},
            {"id": "b", "url": "https://same.test/b", "content": "Also short", "date": "2023-01-01"},
            {"id": "c", "content": "Missing provenance"},
        ],
        now=datetime(2026, 5, 1, tzinfo=timezone.utc),
    )

    assert report["risk_count"] == 4
    assert [risk["type"] for risk in report["risks"]] == [
        "missing-provenance",
        "single-source-dependence",
        "stale-evidence",
        "short-content",
    ]
    assert report["summary"]["source_domains"] == {"same.test": 2, "unknown": 1}


def test_flag_context_risks_detects_conflicting_dates():
    report = flag_context_risks(
        [
            {
                "id": "conflict",
                "source_project": "web",
                "content": "Long enough content " * 10,
                "created_at": "2022-01-01",
                "updated_at": "2025-01-01",
            }
        ],
        now=date(2026, 1, 1),
    )

    assert report["risks"] == [
        {
            "type": "conflicting-dates",
            "severity": "medium",
            "message": "some results contain dates more than a year apart",
            "result_ids": ["conflict"],
        }
    ]


def test_flag_context_risks_accepts_naive_datetime_now():
    report = flag_context_risks(
        [{"id": "a", "source_project": "web", "content": "Long enough content " * 10, "date": "2025-05-01"}],
        now=datetime(2026, 5, 1),
    )

    assert report["risk_count"] == 0


def test_flag_context_risks_validates_now():
    with pytest.raises(ValueError, match="now must be a date, datetime, or None"):
        flag_context_risks([], now="2026-01-01")
