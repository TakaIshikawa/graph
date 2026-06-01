from graph.rag import detect_query_hipaa_requirement


def test_hipaa_phi_baa_and_covered_entity_questions_are_detected():
    report = detect_query_hipaa_requirement("HIPAA BAA for PHI and covered entity workflows.")

    assert report["requires_hipaa"] is True
    assert report["agreement_cues"] == ["BAA"]
    assert report["protected_data_cues"] == ["PHI"]
    assert report["entity_cues"] == ["covered entity"]


def test_ephi_and_minimum_necessary_are_detected():
    report = detect_query_hipaa_requirement("How is ePHI handled under minimum necessary access?")

    assert report["protected_data_cues"] == ["ePHI"]
    assert report["safeguard_cues"] == ["minimum necessary"]


def test_unrelated_privacy_question_does_not_match():
    assert detect_query_hipaa_requirement("Do you publish a privacy policy?")["requires_hipaa"] is False
