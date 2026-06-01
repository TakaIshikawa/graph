from graph.rag import detect_query_soc2_requirement


def test_soc2_and_type_ii_report_queries_are_detected():
    report = detect_query_soc2_requirement("Need the SOC 2 Type II report and audit report.")

    assert report["requires_soc2"] is True
    assert report["report_types"] == ["type_ii", "audit_report"]
    assert report["confidence"] == "high"


def test_trust_services_cues_are_categorized():
    report = detect_query_soc2_requirement("SOC2 trust services criteria for security availability and confidentiality.")

    assert report["trust_service_cues"] == ["trust_services_criteria", "security", "availability", "confidentiality"]


def test_generic_audit_language_without_soc2_does_not_match():
    assert detect_query_soc2_requirement("Send the annual audit report.")["requires_soc2"] is False
