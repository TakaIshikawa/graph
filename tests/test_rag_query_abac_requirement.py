from graph.rag.query_abac_requirement import detect_query_abac_requirements


def test_detects_abac_acronym_and_decision_point():
    result = detect_query_abac_requirements("ABAC requirements need a policy decision point.")

    assert result["has_abac_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["decision_point"]


def test_detects_conditional_access_policy_conditions():
    result = detect_query_abac_requirements("Conditional access policy docs must cover policy conditions.")

    assert [row["category"] for row in result["requirements"]] == ["policy_conditions"]


def test_detects_resource_and_subject_attributes():
    result = detect_query_abac_requirements("Access policy evaluation depends on subject attributes and resource attributes.")

    assert [row["category"] for row in result["requirements"]] == ["resource_attributes", "subject_attributes"]


def test_detects_environment_attributes():
    result = detect_query_abac_requirements("ABAC should evaluate environmental attributes like time of day and network location.")

    assert [row["category"] for row in result["requirements"]] == ["environment_attributes"]


def test_ignores_generic_attribute_data_model_wording_without_access_control_context():
    assert detect_query_abac_requirements("Which customer attributes belong in the data model and reporting schema?") == {
        "has_abac_requirements": False,
        "requirements": [],
    }
