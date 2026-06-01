from graph.rag.query_sustainability_requirement import detect_query_sustainability_requirements


def test_detects_sustainability_categories():
    rows = detect_query_sustainability_requirements(
        "Include carbon footprint, energy usage, environmental impact, green hosting, and resource efficiency."
    )

    assert rows == [
        {"matched_text": "carbon footprint", "category": "carbon_footprint", "severity": "high"},
        {"matched_text": "energy usage", "category": "energy_usage", "severity": "medium"},
        {"matched_text": "environmental impact", "category": "environmental_impact", "severity": "high"},
        {"matched_text": "green hosting", "category": "green_hosting", "severity": "medium"},
        {"matched_text": "resource efficiency", "category": "resource_efficiency", "severity": "medium"},
    ]


def test_handles_mixed_case_and_empty_queries():
    assert detect_query_sustainability_requirements("Minimize POWER   CONSUMPTION.") == [
        {"matched_text": "POWER CONSUMPTION", "category": "energy_usage", "severity": "medium"}
    ]
    assert detect_query_sustainability_requirements("") == []


def test_returns_no_rows_for_unrelated_query():
    assert detect_query_sustainability_requirements("Summarize pricing tiers.") == []
