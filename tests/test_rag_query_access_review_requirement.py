from graph.rag.query_access_review_requirement import detect_query_access_review_requirement


def test_access_review_signals_are_detected():
    report = detect_query_access_review_requirement(
        "Need access review, access recertification, entitlement review, role review, least privilege review, and reviewer signoff."
    )

    assert report["requires_access_review"] is True
    assert report["categories"] == [
        "access_review",
        "access_recertification",
        "entitlement_review",
        "role_review",
        "least_privilege_review",
        "reviewer_signoff",
    ]


def test_review_cadence_is_captured_before_access_review_tie():
    report = detect_query_access_review_requirement("Quarterly access review requires reviewer approval.")

    assert report["categories"] == ["cadence", "access_review", "reviewer_signoff"]
    assert report["matches"][0]["span"] == (0, 23)


def test_unrelated_access_query_returns_empty_result():
    assert detect_query_access_review_requirement("Explain how users access the dashboard.") == {
        "requires_access_review": False,
        "categories": [],
        "matches": [],
    }
