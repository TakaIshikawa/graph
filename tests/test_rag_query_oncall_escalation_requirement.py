from graph.rag import detect_query_oncall_escalation_requirement


def test_oncall_pager_and_escalation_policy_questions_are_detected():
    report = detect_query_oncall_escalation_requirement("Share on-call pager and escalation policy details.")

    assert report["requires_oncall_escalation"] is True
    assert report["cue_categories"] == ["coverage", "pager"]


def test_coverage_and_severity_cues_are_detected():
    report = detect_query_oncall_escalation_requirement("After-hours coverage, response owner, severity levels, P1 and P2.")

    assert report["cue_categories"] == ["coverage", "severity", "owner"]


def test_generic_support_contact_does_not_match():
    assert detect_query_oncall_escalation_requirement("What is the support email address?")["requires_oncall_escalation"] is False
