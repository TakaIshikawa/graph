from graph.rag.query_confidentiality_requirement import detect_query_confidentiality_requirement


def test_detects_confidentiality_signals_in_deterministic_order():
    report = detect_query_confidentiality_requirement("Use confidential NDA material from a private dataset; do not share it.")
    assert report == {
        "requires_confidentiality": True,
        "signals": ["confidential", "nda", "private_dataset", "do_not_share"],
        "redaction_recommended": True,
    }


def test_redaction_terms_recommend_redaction():
    assert detect_query_confidentiality_requirement("Please anonymize and redact names.")["redaction_recommended"] is True


def test_generic_private_preference_does_not_trigger():
    assert detect_query_confidentiality_requirement("I privately prefer shorter answers.") == {
        "requires_confidentiality": False,
        "signals": [],
        "redaction_recommended": False,
    }
