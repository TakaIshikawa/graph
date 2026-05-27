from __future__ import annotations

from graph.store import summarize_unit_timestamp_consistency


def test_timestamp_consistency_counts_missing_invalid_and_ordering_issues():
    summary = summarize_unit_timestamp_consistency(
        [
            {"id": "ok", "metadata": {"created_at": "2024-01-01", "updated_at": "2024-01-02T03:00:00Z", "imported_at": "2024-01-03T00:00:00+00:00"}},
            {"id": "missing-created", "metadata": {"updated_at": "2024-01-02"}},
            {"id": "missing-updated", "metadata": {"created_at": "2024-01-01"}},
            {"id": "bad-updated", "metadata": {"created_at": "2024-01-03", "updated_at": "2024-01-02"}},
            {"id": "bad-imported", "metadata": {"created_at": "2024-01-03T01:00:00+09:00", "updated_at": "2024-01-03T01:30:00+09:00", "imported_at": "2024-01-02T15:00:00Z"}},
            {"id": "invalid", "metadata": {"created_at": "not a date", "updated_at": "2024-01-02"}},
        ],
        sample_limit=2,
    )

    assert summary["total_units"] == 6
    assert summary["issue_counts"] == {
        "missing_created": 1,
        "missing_updated": 1,
        "updated_before_created": 1,
        "imported_before_created": 1,
        "invalid_timestamp": 1,
    }
    assert summary["example_unit_ids"]["missing_created"] == ["missing-created"]
    assert summary["example_unit_ids"]["updated_before_created"] == ["bad-updated"]


def test_timestamp_consistency_limits_examples_deterministically():
    summary = summarize_unit_timestamp_consistency(
        [{"id": f"u{index}", "metadata": {"updated_at": "2024-01-01"}} for index in range(4)],
        sample_limit=2,
    )

    assert summary["missing_created_count"] == 4
    assert summary["example_unit_ids"]["missing_created"] == ["u0", "u1"]
