from graph.rag.query_data_residency_requirement import detect_query_data_residency_requirements


def test_detects_data_residency_categories():
    rows = detect_query_data_residency_requirements(
        "Require data residency, regional hosting, sovereign cloud, EU-only, US-only, cross-border transfer, and jurisdictional storage."
    )

    assert rows == [
        {"matched_text": "cross-border transfer", "category": "cross_border_transfer", "severity": "high"},
        {"matched_text": "data residency", "category": "data_residency", "severity": "high"},
        {"matched_text": "EU-only", "category": "eu_only", "severity": "high"},
        {"matched_text": "jurisdictional storage", "category": "jurisdictional_storage", "severity": "high"},
        {"matched_text": "regional hosting", "category": "regional_hosting", "severity": "medium"},
        {"matched_text": "sovereign cloud", "category": "sovereign_cloud", "severity": "high"},
        {"matched_text": "US-only", "category": "us_only", "severity": "high"},
    ]


def test_handles_explicit_region_wording():
    assert detect_query_data_residency_requirements("Keep records WITHIN   THE   EU.") == [
        {"matched_text": "WITHIN THE EU", "category": "eu_only", "severity": "high"}
    ]


def test_unrelated_geography_does_not_match():
    assert detect_query_data_residency_requirements("Compare adoption in Europe and Asia.") == []
