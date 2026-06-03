from graph.rag import detect_query_sox_requirement


def test_sox_requirement_detects_control_categories():
    report = detect_query_sox_requirement(
        "SOX Section 404 ICFR evidence for segregation of duties, audit evidence, and change control for financial systems."
    )

    assert report["requires_sox"] is True
    assert report["categories"] == [
        "audit_evidence",
        "financial_change_control",
        "icfr",
        "section_404",
        "segregation_of_duties",
        "sox",
    ]
    assert report["matches"][0]["matched_text"] == "SOX"
    assert {"matched_text", "category", "severity", "span"} <= report["matches"][0].keys()


def test_sox_requirement_ignores_socks_and_generic_audit():
    report = detect_query_sox_requirement("Find socks and summarize generic audit phrasing in finance news.")

    assert report["requires_sox"] is False
    assert report["matches"] == []
