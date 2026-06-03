from graph.rag.query_deprecation_policy_requirement import detect_query_deprecation_policy_requirements


def test_detects_api_deprecation_policy_categories():
    result = detect_query_deprecation_policy_requirements(
        "For the API deprecation policy, include a deprecation notice, sunset date, migration guide, "
        "version support window, and backward compatibility expectations."
    )

    assert result["has_deprecation_policy_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "notice_period",
        "sunset_date",
        "migration_guide",
        "version_support",
        "backward_compatibility",
    ]


def test_detects_platform_policy_with_multiple_categories():
    result = detect_query_deprecation_policy_requirements(
        "When the platform retires v1 endpoints, what advance notice, removal date, transition guide, "
        "supported versions, and legacy clients policy apply?"
    )

    assert result["has_deprecation_policy_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == [
        "notice_period",
        "sunset_date",
        "migration_guide",
        "version_support",
        "backward_compatibility",
    ]


def test_context_gating_avoids_slang_or_generic_old_content():
    slang = detect_query_deprecation_policy_requirements("That deprecated slang is old; explain the phrase sunset.")
    generic = detect_query_deprecation_policy_requirements("Summarize old content and obsolete notes in this article.")

    assert slang["has_deprecation_policy_requirements"] is False
    assert slang["requirements"] == []
    assert generic["has_deprecation_policy_requirements"] is False
    assert generic["requirements"] == []


def test_deprecation_without_policy_categories_does_not_require_policy_details():
    result = detect_query_deprecation_policy_requirements("Which API endpoints are deprecated?")

    assert result["has_deprecation_policy_requirements"] is False
    assert result["requirements"] == []
