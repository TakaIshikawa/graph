from graph.rag.query_data_classification_requirement import detect_query_data_classification_requirement


def test_data_classification_categories_are_detected():
    report = detect_query_data_classification_requirement(
        "Require data classification for confidential data, regulated data handling, and classification labels."
    )

    assert report["requires_data_classification"] is True
    assert report["categories"] == ["classification_scheme", "sensitive_data", "regulated_data", "handling_label"]
    assert report["matches"][1]["matched_text"] == "confidential data"


def test_pii_sensitivity_and_restricted_data_are_sensitive_data():
    report = detect_query_data_classification_requirement("Check PII sensitivity and restricted data handling.")

    assert report["categories"] == ["sensitive_data"]
    assert report["matches"][0]["span"] == (6, 21)


def test_generic_privacy_wording_does_not_trigger():
    assert detect_query_data_classification_requirement("Compare privacy controls for user profiles.") == {
        "requires_data_classification": False,
        "categories": [],
        "matches": [],
    }
