from graph.rag.query_incident_response_requirement import detect_query_incident_response_requirement


def test_detects_incident_lifecycle_signals():
    report = detect_query_incident_response_requirement(
        "Build an incident response playbook with triage, SEV-1 classification, "
        "containment, remediation, and a postmortem."
    )

    assert report == {
        "requires_incident_response": True,
        "signals": ["incident_response", "triage", "severity", "containment", "remediation", "postmortem"],
        "customer_communication_required": False,
    }


def test_customer_communication_and_status_page_set_flag():
    report = detect_query_incident_response_requirement(
        "Include customer communications, notify users, and status page updates during mitigation."
    )

    assert report == {
        "requires_incident_response": True,
        "signals": ["remediation", "communications", "status_page"],
        "customer_communication_required": True,
    }


def test_repeated_lifecycle_terms_are_deduplicated():
    report = detect_query_incident_response_requirement(
        "Triage, triage, containment, contain the issue, remediation, mitigation, post-incident review."
    )

    assert report["signals"] == ["triage", "containment", "remediation", "postmortem"]


def test_unrelated_query_has_no_incident_response_requirement():
    assert detect_query_incident_response_requirement("Summarize the release notes and migration guide.") == {
        "requires_incident_response": False,
        "signals": [],
        "customer_communication_required": False,
    }
