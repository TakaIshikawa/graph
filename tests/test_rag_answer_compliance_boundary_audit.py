from graph.rag.answer_compliance_boundary_audit import audit_answer_compliance_boundaries


def test_detects_regulated_domains_from_query_and_answer():
    report = audit_answer_compliance_boundaries("This contract may create tax liability.", query="Legal and financial impact?")
    assert report["regulated_domain"] is True
    assert report["domains"] == ["legal", "tax", "financial"]
    assert report["boundary_present"] is False
    assert report["recommendation"] == "add_compliance_boundary_language"


def test_detects_boundary_language_case_insensitively():
    report = audit_answer_compliance_boundaries("This is NOT LEGAL ADVICE; consult qualified counsel.", query="Contract compliance")
    assert report["boundary_present"] is True
    assert report["recommendation"] == ""
