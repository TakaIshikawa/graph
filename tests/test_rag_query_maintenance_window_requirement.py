from graph.rag.query_maintenance_window_requirement import detect_query_maintenance_window_requirement


def test_scheduled_downtime_and_time_window_are_captured():
    report = detect_query_maintenance_window_requirement("Use a maintenance window for scheduled downtime 1am-3am.")

    assert report["requires_maintenance_window"] is True
    assert report["window_terms"] == ["maintenance_window", "scheduled_downtime"]
    assert report["time_windows"] == ["1am-3am"]
    assert report["confidence"] == "high"


def test_blackout_freeze_and_no_downtime_are_distinguishable():
    report = detect_query_maintenance_window_requirement("Respect the blackout period, freeze window, and no-downtime maintenance.")

    assert report["window_terms"] == ["blackout_period", "freeze_window", "no_downtime"]
    assert report["confidence"] == "medium"


def test_unrelated_scheduling_question_does_not_trigger():
    assert detect_query_maintenance_window_requirement("Schedule a meeting next week.") == {
        "requires_maintenance_window": False,
        "window_terms": [],
        "time_windows": [],
        "matched_phrases": [],
        "recommendations": [],
        "confidence": "none",
    }
