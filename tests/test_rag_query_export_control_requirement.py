from graph.rag.query_export_control_requirement import detect_query_export_control_requirement


def test_extracts_named_export_control_frameworks_case_insensitively():
    result = detect_query_export_control_requirement("Check ear, ITAR, and OfAc constraints before answering.")

    assert result["requires_export_control_review"] is True
    assert result["frameworks"] == ["ear", "itar", "ofac"]
    assert result["severity"] == "high"


def test_sanctions_cues_trigger_review_without_framework():
    result = detect_query_export_control_requirement("Require denied-party screening for embargoed countries and dual-use data.")

    assert result["requires_export_control_review"] is True
    assert result["frameworks"] == []
    assert [cue["category"] for cue in result["matched_cues"]] == ["embargo", "denied_party", "dual_use"]
    assert result["severity"] == "high"


def test_unrelated_international_query_returns_defaults():
    assert detect_query_export_control_requirement("Compare international launch timelines by region.") == {
        "requires_export_control_review": False,
        "frameworks": [],
        "matched_cues": [],
        "severity": "none",
    }
