from graph.rag.query_backup_recovery_requirement import detect_backup_recovery_requirement


def test_detects_backup_restore_rpo_and_rto_requirements():
    result = detect_backup_recovery_requirement(
        "Need a backup policy, restore from backup procedure, RPO under 15 minutes, and recovery time objective below 4 hours."
    )

    assert result["has_backup_recovery_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == ["backup_policy", "restore", "rpo", "rto"]
    assert {row["severity"] for row in result["requirements"] if row["category"] in {"restore", "rpo", "rto"}} == {"high"}


def test_detects_point_in_time_restore_and_snapshot_retention():
    result = detect_backup_recovery_requirement("Compare point-in-time restore support and snapshot retention windows.")

    assert [row["category"] for row in result["requirements"]] == ["point_in_time_restore", "snapshot_retention"]


def test_detects_disaster_recovery_phrases():
    result = detect_backup_recovery_requirement("Include disaster recovery evidence and the DR plan requirements.")

    assert [row["category"] for row in result["requirements"]] == ["disaster_recovery"]
    assert result["requirements"][0]["severity"] == "high"


def test_generic_save_and_export_queries_remain_negative():
    assert detect_backup_recovery_requirement("Save this report and export the CSV results.") == {
        "has_backup_recovery_requirement": False,
        "requirements": [],
    }
