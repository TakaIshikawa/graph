from graph.rag.query_time_horizon import detect_query_time_horizon


def test_detects_relative_horizon():
    report = detect_query_time_horizon("Show incidents from the last 30 days.")

    assert report["horizon_type"] == "relative"
    assert report["normalized_value"] == "30 days"
    assert report["requires_fresh_sources"] is True


def test_detects_absolute_horizon():
    report = detect_query_time_horizon("Compare results since 2020.")

    assert report["horizon_type"] == "absolute"
    assert report["normalized_value"] == "since 2020"


def test_detects_freshness_oriented_query():
    report = detect_query_time_horizon("Find the latest guidance.")

    assert report["horizon_type"] == "freshness"
    assert report["requires_fresh_sources"] is True


def test_detects_historical_query():
    report = detect_query_time_horizon("Give historical context.")

    assert report["horizon_type"] == "historical"


def test_absent_horizon_is_stable():
    assert detect_query_time_horizon("Explain the concept.") == {
        "horizon_type": "absent",
        "normalized_value": None,
        "cues": [],
        "requires_fresh_sources": False,
        "confidence": 0.0,
    }
