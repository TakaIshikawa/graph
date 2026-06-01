from graph.rag.query_chain_of_custody_requirement import detect_query_chain_of_custody_requirement


def test_chain_of_custody_and_records_trigger():
    report = detect_query_chain_of_custody_requirement("Maintain chain-of-custody with custody records.")

    assert report == {
        "requires_chain_of_custody": True,
        "custody_controls": ["chain_of_custody", "custody_records"],
        "matched_cues": ["chain_of_custody", "custody_records"],
        "severity": "high",
    }


def test_tamper_evident_logs_are_included():
    report = detect_query_chain_of_custody_requirement("Use tamper-evident logs for evidence handling.")

    assert report["custody_controls"] == ["evidence_handling", "tamper_evident_logs"]
    assert report["severity"] == "high"


def test_generic_audit_logs_do_not_trigger():
    assert detect_query_chain_of_custody_requirement("Add audit logs for admin changes.") == {
        "requires_chain_of_custody": False,
        "custody_controls": [],
        "matched_cues": [],
        "severity": "none",
    }
