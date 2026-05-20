from __future__ import annotations

from graph.rag.temporal_consistency import analyze_temporal_consistency


def test_temporal_consistency_marks_in_window_evidence_from_parameters():
    summary = analyze_temporal_consistency(
        "revenue changes",
        [
            {"id": "a", "published_at": "2024-02-15T10:30:00Z", "source": "filing"},
            {"id": "b", "metadata": {"start_date": "2024-03-01", "end_date": "2024-03-31"}},
        ],
        reference_date="2024-04-01",
        window_start="2024-01-01",
        window_end="2024-12-31",
    )

    assert summary["window"] == {"start": "2024-01-01", "end": "2024-12-31", "source": "parameters"}
    assert summary["counts"]["in_window"] == 2
    assert summary["date_buckets"]["in_window"] == 2
    assert summary["result_ids"]["in_window"] == ["a", "b"]
    assert summary["warnings"] == []


def test_temporal_consistency_flags_stale_evidence_without_window():
    summary = analyze_temporal_consistency(
        "latest policy",
        [{"id": "old", "updated_at": "2020-01-01"}],
        reference_date="2024-01-01",
        stale_after_days=365,
    )

    assert summary["counts"]["stale"] == 1
    assert summary["date_buckets"]["stale"] == 1
    assert summary["warnings"] == ["stale_evidence"]


def test_temporal_consistency_flags_future_dated_evidence():
    summary = analyze_temporal_consistency(
        "current roadmap",
        [{"id": "future", "created_at": "2025-01-02"}],
        reference_date="2025-01-01",
    )

    assert summary["counts"]["future"] == 1
    assert summary["date_buckets"]["future"] == 1
    assert summary["warnings"] == ["future_dated_evidence"]


def test_temporal_consistency_flags_undated_evidence():
    summary = analyze_temporal_consistency("current roadmap", [{"id": "missing"}], reference_date="2025-01-01")

    assert summary["counts"]["missing_date"] == 1
    assert summary["date_buckets"]["missing_date"] == 1
    assert summary["warnings"] == ["missing_date_evidence"]


def test_temporal_consistency_infers_query_year_and_warning_order_is_stable():
    summary = analyze_temporal_consistency(
        "incidents during 2024",
        [
            {"id": "missing"},
            {"id": "future", "date": "2026-01-01"},
            {"id": "before", "date": "2023-12-01"},
            {"id": "inside", "date": "2024-06-01"},
        ],
        reference_date="2025-01-01",
    )

    assert summary["window"] == {"start": "2024-01-01", "end": "2024-12-31", "source": "query"}
    assert summary["counts"]["out_of_window"] == 2
    assert summary["result_ids"]["before_window"] == ["before"]
    assert summary["result_ids"]["future"] == ["future"]
    assert summary["warnings"] == [
        "future_dated_evidence",
        "missing_date_evidence",
        "out_of_window_evidence",
        "stale_evidence",
    ]
