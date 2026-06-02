from graph.rag.query_backup_retention_requirement import detect_query_backup_retention_requirement


def test_detects_combined_disaster_recovery_query():
    result = detect_query_backup_retention_requirement(
        "Need DR evidence for backup retention, restore testing, snapshots, and point-in-time recovery."
    )

    assert result["has_backup_retention_requirement"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "backup_retention",
        "point_in_time_recovery",
        "restore_testing",
        "snapshot",
    ]
    assert _severity(result, "restore_testing") == "high"
    assert _severity(result, "point_in_time_recovery") == "high"


def test_detects_retention_wording():
    result = detect_query_backup_retention_requirement("Must retain backups for 35 days with a retention window.")

    assert [row["category"] for row in result["requirements"]] == ["backup_retention"]
    assert result["requirements"][0]["severity"] == "medium"


def test_detects_rpo_rto_wording_as_high_impact():
    result = detect_query_backup_retention_requirement(
        "Compare vendors with RPO under 15 minutes and recovery time objective below 4 hours."
    )

    assert [row["category"] for row in result["requirements"]] == ["rpo", "rto"]
    assert {row["severity"] for row in result["requirements"]} == {"high"}


def test_avoids_unrelated_file_backup_mentions():
    result = detect_query_backup_retention_requirement("Where did I put the backup file for the migration notes?")

    assert result == {"has_backup_retention_requirement": False, "requirements": []}


def _severity(result, category):
    return next(row["severity"] for row in result["requirements"] if row["category"] == category)
