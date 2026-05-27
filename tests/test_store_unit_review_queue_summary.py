from __future__ import annotations

from graph.store.unit_review_queue_summary import summarize_unit_review_queue


def test_review_queue_summary_groups_by_source_and_counts_review_signals():
    summary = summarize_unit_review_queue(
        [
            {
                "id": "a",
                "source_project": "docs",
                "metadata": {
                    "review_status": "needs_review",
                    "review_due": "2026-05-20T00:00:00Z",
                    "priority": "urgent",
                },
            },
            {
                "id": "b",
                "source_project": "docs",
                "metadata": {
                    "status": "Blocked",
                    "due_at": "2026-05-25T00:00:00+00:00",
                    "review_priority": 9,
                },
            },
            {
                "id": "c",
                "metadata": {
                    "source": "inbox",
                    "triage_status": "open",
                    "next_review_at": "2026-05-30T00:00:00Z",
                    "priority": "low",
                },
            },
        ],
        now="2026-05-24T00:00:00+00:00",
    )

    assert summary == {
        "total_units": 3,
        "rows": [
            {
                "source": "docs",
                "unit_count": 2,
                "review_requested_count": 1,
                "overdue_count": 1,
                "high_priority_count": 2,
                "blocked_count": 1,
                "next_due_unit_id": "a",
            },
            {
                "source": "inbox",
                "unit_count": 1,
                "review_requested_count": 0,
                "overdue_count": 0,
                "high_priority_count": 0,
                "blocked_count": 0,
                "next_due_unit_id": "c",
            },
        ],
        "source_summaries": [
            {
                "source": "docs",
                "unit_count": 2,
                "review_requested_count": 1,
                "overdue_count": 1,
                "high_priority_count": 2,
                "blocked_count": 1,
                "next_due_unit_id": "a",
            },
            {
                "source": "inbox",
                "unit_count": 1,
                "review_requested_count": 0,
                "overdue_count": 0,
                "high_priority_count": 0,
                "blocked_count": 0,
                "next_due_unit_id": "c",
            },
        ],
    }


def test_review_queue_summary_handles_numeric_textual_priority_blocked_case_and_bad_dates():
    summary = summarize_unit_review_queue(
        [
            {"id": "low-num", "metadata": {"source": "s", "priority": 3, "review_due": "not-a-date"}},
            {"id": "high-num", "metadata": {"source": "s", "priority": "8"}},
            {"id": "high-text", "metadata": {"source": "s", "review_priority": "P1", "status": "BLOCKED"}},
        ],
        now="2026-05-24T00:00:00+00:00",
    )

    row = summary["rows"][0]
    assert row["high_priority_count"] == 2
    assert row["blocked_count"] == 1
    assert row["overdue_count"] == 0
    assert row["next_due_unit_id"] == ""
