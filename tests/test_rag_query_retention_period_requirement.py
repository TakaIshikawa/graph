from graph.rag.query_retention_period_requirement import detect_query_retention_period_requirements


def test_detects_retention_periods_and_durations_deterministically():
    result = detect_query_retention_period_requirements("Keep records for 30 days, then retain for 7 years for reports.")

    assert result == {
        "has_retention_period_requirements": True,
        "requirements": ["retention_period"],
        "explicit_duration_mentions": ["30 days", "7 years"],
        "legal_retention_sensitive": False,
    }


def test_detects_deletion_archive_purge_and_records_categories():
    result = detect_query_retention_period_requirements(
        "Define the records retention policy, deletion window, archive data for 12 months, and purge schedule."
    )

    assert result["has_retention_period_requirements"] is True
    assert result["requirements"] == ["deletion_window", "archival_period", "purge_schedule", "records_retention"]
    assert result["explicit_duration_mentions"] == ["12 months"]


def test_legal_retention_sets_sensitive_flag():
    result = detect_query_retention_period_requirements("What statutory retention period applies for 7 years?")

    assert result["has_retention_period_requirements"] is True
    assert result["requirements"] == ["retention_period", "legal_retention"]
    assert result["explicit_duration_mentions"] == ["7 years"]
    assert result["legal_retention_sensitive"] is True


def test_unrelated_query_returns_empty_results():
    assert detect_query_retention_period_requirements("Show usage growth for the last 30 days.") == {
        "has_retention_period_requirements": False,
        "requirements": [],
        "explicit_duration_mentions": [],
        "legal_retention_sensitive": False,
    }
