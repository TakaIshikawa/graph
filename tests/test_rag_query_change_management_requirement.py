from graph.rag.query_change_management_requirement import detect_query_change_management_requirement


def test_change_management_signals_are_detected_in_first_occurrence_order():
    report = detect_query_change_management_requirement(
        "Require CAB approval before a change ticket during a change freeze for production change control."
    )

    assert report["requires_change_management"] is True
    assert report["categories"] == ["cab_approval", "change_ticket", "change_freeze", "production_change_control"]
    assert report["matches"][0]["matched_text"] == "CAB approval"
    assert report["matches"][0]["span"] == (8, 20)


def test_emergency_change_and_change_management_are_detected():
    report = detect_query_change_management_requirement("Document change management for each emergency change.")

    assert report["categories"] == ["change_management", "emergency_change"]
    assert [match["severity"] for match in report["matches"]] == ["medium", "high"]


def test_unrelated_query_returns_empty_result():
    assert detect_query_change_management_requirement("Summarize release notes for the new feature.") == {
        "requires_change_management": False,
        "categories": [],
        "matches": [],
    }
