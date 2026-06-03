from graph.rag import detect_query_coppa_requirement


def test_coppa_requirement_detects_child_privacy_categories():
    report = detect_query_coppa_requirement(
        "COPPA review for users under 13, verifiable parental consent, parental notice, child-directed app, and delete children's data."
    )

    assert report["requires_coppa"] is True
    assert report["categories"] == [
        "child_directed_service",
        "coppa",
        "deletion_rights",
        "parental_notice",
        "under_13",
        "verifiable_parental_consent",
    ]
    assert report["matches"][0]["category"] == "child_directed_service"
    assert {"matched_text", "category", "severity", "span"} <= report["matches"][0].keys()


def test_coppa_requirement_ignores_generic_child_wording():
    report = detect_query_coppa_requirement("Ideas for parenting content with a child theme and age 8 examples.")

    assert report["requires_coppa"] is False
    assert report["matches"] == []
