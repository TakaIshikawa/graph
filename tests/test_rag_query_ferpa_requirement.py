from graph.rag import detect_query_ferpa_requirement


def test_ferpa_requirement_detects_education_privacy_categories():
    report = detect_query_ferpa_requirement(
        "FERPA controls for education records, student PII, directory information, school official exception, and access and amendment."
    )

    assert report["requires_ferpa"] is True
    assert report["categories"] == [
        "access_amendment",
        "consent",
        "directory_information",
        "education_records",
        "ferpa",
        "student_pii",
    ]
    assert report["matches"][0]["matched_text"] == "FERPA"
    assert {"matched_text", "category", "severity", "span"} <= report["matches"][0].keys()


def test_ferpa_requirement_ignores_generic_school_queries():
    report = detect_query_ferpa_requirement("Compare school scheduling tools for student learning content.")

    assert report["requires_ferpa"] is False
    assert report["matches"] == []
