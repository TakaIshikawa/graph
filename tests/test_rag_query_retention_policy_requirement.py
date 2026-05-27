from __future__ import annotations

from graph.rag.query_retention_policy_requirement import detect_query_retention_policy_requirement


def test_detect_query_retention_policy_requirement_extracts_explicit_windows():
    result = detect_query_retention_policy_requirement("What is the data retention policy for records kept for 30 days?")

    assert result == {
        "requires_retention_policy": True,
        "retention_terms": ["data_retention"],
        "time_windows": ["30 days"],
        "matched_phrases": ["data retention", "kept"],
    }


def test_detect_query_retention_policy_requirement_detects_deletion_requests():
    result = detect_query_retention_policy_requirement("Do users have a right to delete or erase account data after 7 years?")

    assert result["requires_retention_policy"] is True
    assert result["retention_terms"] == ["deletion_window"]
    assert result["time_windows"] == ["7 years"]
    assert result["matched_phrases"] == ["right to delete", "erase"]


def test_detect_query_retention_policy_requirement_detects_log_retention():
    result = detect_query_retention_policy_requirement("How long are audit logs retained for investigations?")

    assert result["requires_retention_policy"] is True
    assert result["retention_terms"] == ["data_retention", "logs"]
    assert result["matched_phrases"] == ["logs", "retained"]


def test_detect_query_retention_policy_requirement_detects_backup_retention():
    result = detect_query_retention_policy_requirement("Are backups purged after 90 days or archived for 1 year?")

    assert result["requires_retention_policy"] is True
    assert result["retention_terms"] == ["archival_period", "backups", "deletion_window"]
    assert result["time_windows"] == ["90 days", "1 year"]


def test_detect_query_retention_policy_requirement_ignores_unrelated_temporal_queries():
    result = detect_query_retention_policy_requirement("Show revenue growth over the last 30 days.")

    assert result == {
        "requires_retention_policy": False,
        "retention_terms": [],
        "time_windows": [],
        "matched_phrases": [],
    }
