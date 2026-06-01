from graph.rag.query_legal_hold_requirement import detect_query_legal_hold_requirement


def test_litigation_hold_triggers_legal_hold():
    assert detect_query_legal_hold_requirement("Apply a litigation hold and preserve evidence.") == {
        "requires_legal_hold": True,
        "preservation_actions": ["legal_hold", "preserve_evidence"],
        "matched_cues": ["legal_hold", "preserve_evidence"],
        "severity": "high",
    }


def test_ediscovery_and_freeze_deletion_are_reported():
    report = detect_query_legal_hold_requirement("Preserve for eDiscovery and freeze deletion of records.")

    assert report["preservation_actions"] == ["ediscovery", "retention_freeze"]
    assert report["severity"] == "high"


def test_routine_retention_policy_does_not_trigger():
    assert detect_query_legal_hold_requirement("Summarize the retention policy for old logs.") == {
        "requires_legal_hold": False,
        "preservation_actions": [],
        "matched_cues": [],
        "severity": "none",
    }
