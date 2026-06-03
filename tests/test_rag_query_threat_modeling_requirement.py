from graph.rag.query_threat_modeling_requirement import detect_query_threat_modeling_requirements


def test_detects_threat_modeling_requirement_categories_in_stable_order():
    result = detect_query_threat_modeling_requirements(
        "Build a threat model with STRIDE, assets, trust boundary DFD, threats, mitigation mapping, and review cadence."
    )

    assert result["has_threat_modeling_requirements"] is True
    assert result["rows"] == [
        {"category": "methodology", "matched_text": "threat model", "severity": "high"},
        {"category": "assets", "matched_text": "assets", "severity": "medium"},
        {"category": "trust_boundaries", "matched_text": "trust boundary", "severity": "high"},
        {"category": "threats", "matched_text": "threats", "severity": "high"},
        {"category": "mitigations", "matched_text": "mitigation mapping", "severity": "high"},
        {"category": "review_cadence", "matched_text": "review cadence", "severity": "medium"},
    ]


def test_recognizes_attack_tree_misuse_case_abuse_case_and_data_flow_diagram_terms():
    result = detect_query_threat_modeling_requirements(
        "Compare attack tree, misuse case, abuse case, and data flow diagram evidence."
    )

    assert result["rows"] == [
        {"category": "methodology", "matched_text": "attack tree", "severity": "high"},
        {"category": "trust_boundaries", "matched_text": "data flow diagram", "severity": "high"},
    ]


def test_requires_threat_modeling_context_before_emitting_categories():
    assert detect_query_threat_modeling_requirements(
        "Inventory assets, controls, threats, mitigations, and quarterly review dates."
    ) == {
        "has_threat_modeling_requirements": False,
        "rows": [],
    }


def test_deduplicates_category_and_keeps_first_match():
    result = detect_query_threat_modeling_requirements(
        "STRIDE and attack trees should map threats to controls with threat model reviews."
    )

    assert result["rows"] == [
        {"category": "methodology", "matched_text": "STRIDE", "severity": "high"},
        {"category": "mitigations", "matched_text": "map threats to controls", "severity": "high"},
        {"category": "threats", "matched_text": "threats", "severity": "high"},
        {"category": "review_cadence", "matched_text": "threat model reviews", "severity": "medium"},
    ]
