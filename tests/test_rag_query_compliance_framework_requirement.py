from graph.rag.query_compliance_framework_requirement import detect_query_compliance_framework_requirements


def test_detects_multiple_compliance_frameworks_deterministically():
    rows = detect_query_compliance_framework_requirements("Need GDPR, SOC 2, HIPAA, and PCI-DSS evidence.")

    assert rows == [
        {"matched_text": "GDPR", "framework": "gdpr", "severity": "high"},
        {"matched_text": "HIPAA", "framework": "hipaa", "severity": "high"},
        {"matched_text": "PCI-DSS", "framework": "pci_dss", "severity": "high"},
        {"matched_text": "SOC 2", "framework": "soc_2", "severity": "high"},
    ]


def test_detects_acronym_and_punctuation_variants():
    rows = detect_query_compliance_framework_requirements("Check ISO/IEC 27001, ISO-27001, Fed RAMP, and CCPA.")

    assert rows == [
        {"matched_text": "CCPA", "framework": "ccpa", "severity": "high"},
        {"matched_text": "Fed RAMP", "framework": "fedramp", "severity": "high"},
        {"matched_text": "ISO/IEC 27001", "framework": "iso_27001", "severity": "high"},
    ]


def test_compliance_framework_no_match_returns_empty_list():
    assert detect_query_compliance_framework_requirements("Show current architecture diagrams.") == []
