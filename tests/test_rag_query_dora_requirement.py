from graph.rag import detect_query_dora_requirement


def test_dora_requirement_detects_eu_resilience_categories():
    report = detect_query_dora_requirement(
        "EU DORA compliance for financial entities, ICT third-party risk, operational resilience testing, ICT incident reporting, register of information, and critical ICT third-party providers."
    )

    assert report["requires_dora"] is True
    assert report["categories"] == [
        "critical_provider",
        "dora",
        "financial_entity",
        "ict_third_party_risk",
        "incident_reporting",
        "register_of_information",
        "resilience_testing",
    ]
    assert report["matches"][0]["matched_text"] == "EU DORA"
    assert {"matched_text", "category", "severity", "span"} <= report["matches"][0].keys()


def test_dora_requirement_ignores_name_and_generic_resilience():
    report = detect_query_dora_requirement("Dora wants an exploration plan for generic resilience wording.")

    assert report["requires_dora"] is False
    assert report["matches"] == []
