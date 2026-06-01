from graph.rag import detect_query_gdpr_requirement


def test_gdpr_and_eu_personal_data_queries_are_detected():
    assert detect_query_gdpr_requirement("GDPR handling for EU personal data.")["requires_gdpr"] is True


def test_gdpr_cue_categories_are_identified():
    report = detect_query_gdpr_requirement("GDPR DSR right to erasure, lawful basis, controller processor, SCCs, and DPIA.")

    assert report["cue_categories"] == ["rights", "lawful_basis", "controller_processor", "transfer", "dpia"]


def test_generic_privacy_language_does_not_match():
    assert detect_query_gdpr_requirement("Explain privacy safeguards.")["requires_gdpr"] is False
