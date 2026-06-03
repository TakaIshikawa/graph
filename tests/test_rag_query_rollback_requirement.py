from __future__ import annotations

from graph.rag.query_rollback_requirement import detect_query_rollback_requirements


def test_operational_rollback_and_revert_requirements_are_detected():
    rows = detect_query_rollback_requirements("Create a rollback plan and explain how to revert the deployment safely.")

    assert [row["requirement"] for row in rows] == ["rollback", "revert"]
    assert rows[0]["matched_text"] == "rollback"
    assert rows[0]["severity"] == "high"
    assert rows[1]["matched_text"] == "revert the deployment"


def test_data_restore_and_disaster_recovery_requirements_are_detected():
    rows = detect_query_rollback_requirements("Include data restore from backup, disaster recovery, RTO, and RPO evidence.")

    assert [row["requirement"] for row in rows] == ["restore", "disaster_recovery"]
    assert rows[0]["matched_text"] == "data restore"
    assert rows[1]["matched_text"] == "disaster recovery"


def test_migration_reversal_and_feature_flag_fallback_are_detected():
    rows = detect_query_rollback_requirements("For the migration rollback, document the feature flag fallback and failover path.")

    assert [row["requirement"] for row in rows] == ["migration_reversal", "feature_flag_fallback", "fallback"]
    assert {row["matched_text"] for row in rows} == {"migration rollback", "feature flag fallback", "failover"}


def test_unrelated_version_control_wording_is_not_flagged():
    assert detect_query_rollback_requirements("How do I revert a Git commit on a feature branch?") == []
    assert detect_query_rollback_requirements("Summarize rollback wording in a source-control guide.") == []
