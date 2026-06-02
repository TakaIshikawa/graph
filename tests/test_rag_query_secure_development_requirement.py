from graph.rag.query_secure_development_requirement import detect_query_secure_development_requirement


def test_secure_development_categories_are_detected():
    report = detect_query_secure_development_requirement(
        "Require secure SDLC, code review, SAST, DAST, dependency scanning, threat modeling, and release security gate."
    )

    assert report["requires_secure_development"] is True
    assert report["categories"] == [
        "review",
        "static_analysis",
        "dynamic_analysis",
        "dependency_scanning",
        "threat_modeling",
        "release_gate",
    ]


def test_common_variants_are_detected():
    report = detect_query_secure_development_requirement("S-SDLC requires software supply chain scanning.")

    assert report["categories"] == ["review", "dependency_scanning"]


def test_unrelated_release_query_returns_empty_result():
    assert detect_query_secure_development_requirement("Summarize release timing for the roadmap.") == {
        "requires_secure_development": False,
        "categories": [],
        "matches": [],
    }
