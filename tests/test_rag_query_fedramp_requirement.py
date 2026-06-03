from graph.rag import detect_query_fedramp_requirement


def test_fedramp_requirement_detects_authorization_categories():
    report = detect_query_fedramp_requirement(
        "FedRAMP agency ATO with JAB authorization, moderate baseline, 3PAO assessment, POA&M, and continuous monitoring."
    )

    assert report["requires_fedramp"] is True
    assert report["categories"] == ["3pao", "ato", "baseline", "continuous_monitoring", "fedramp", "jab", "poam"]
    assert report["matches"][0]["matched_text"] == "FedRAMP"
    assert {"matched_text", "category", "severity", "span"} <= report["matches"][0].keys()


def test_fedramp_requirement_ignores_generic_federal_cloud_words():
    report = detect_query_fedramp_requirement("Plan a ramp-up after federal holidays for cloud hosting.")

    assert report["requires_fedramp"] is False
    assert report["matches"] == []
