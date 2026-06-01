from graph.rag.answer_fallback_path_audit import audit_answer_fallback_paths


def test_complete_fallback_coverage_detects_all_categories():
    summary = audit_answer_fallback_paths(
        "Use fallback failover, degraded mode, a manual workaround, retry failed jobs, and rollback releases."
    )

    assert summary["fallback_paths"] == ["degraded_mode", "fallback", "manual_workaround", "retry", "rollback"]
    assert summary["missing_recommended_paths"] == []
    assert summary["has_operational_fallback"] is True


def test_answer_recommending_actions_without_fallback_paths_reports_missing():
    summary = audit_answer_fallback_paths("Deploy the new index and monitor latency.")

    assert summary["fallback_paths"] == []
    assert summary["missing_recommended_paths"] == ["degraded_mode", "fallback", "manual_workaround", "retry", "rollback"]
    assert summary["has_operational_fallback"] is False
