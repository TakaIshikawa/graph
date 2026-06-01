from graph.rag.query_backup_frequency_requirement import detect_query_backup_frequency_requirement


def test_detects_backup_cadence_terms():
    result = detect_query_backup_frequency_requirement("Need hourly snapshots and weekly backup exports.")

    assert result["requires_backup_frequency"] is True
    assert result["signals"] == ["hourly", "snapshot", "weekly"]
    assert result["cadence_terms"] == ["hourly", "weekly"]


def test_detects_rpo_and_point_in_time_cues():
    result = detect_query_backup_frequency_requirement("Require RPO under 15 minutes with point-in-time recovery.")

    assert result["signals"] == ["point_in_time", "rpo"]
    assert result["cadence_terms"] == []


def test_unrelated_query_returns_empty_output():
    assert detect_query_backup_frequency_requirement("Compare search relevance.") == {
        "requires_backup_frequency": False,
        "signals": [],
        "cadence_terms": [],
    }
