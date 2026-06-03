from graph.rag.query_cors_requirement import detect_query_cors_requirements


def test_detects_cors_headers_for_api():
    result = detect_query_cors_requirements(
        "API CORS policy must define Access-Control-Allow-Origin, Access-Control-Allow-Methods, and Access-Control-Allow-Headers."
    )

    assert result["has_cors_requirements"] is True
    assert result["requirements"] == [
        {"category": "allowed_headers", "matched_text": "Access-Control-Allow-Headers", "severity": "medium"},
        {"category": "allowed_methods", "matched_text": "Access-Control-Allow-Methods", "severity": "medium"},
        {"category": "allowed_origins", "matched_text": "Access-Control-Allow-Origin", "severity": "high"},
    ]


def test_detects_credentialed_requests_and_exposed_headers():
    result = detect_query_cors_requirements(
        "Browser cross-origin fetch requests need with credentials and Access-Control-Expose-Headers."
    )

    assert result["has_cors_requirements"] is True
    assert result["requirements"] == [
        {"category": "credentials", "matched_text": "with credentials", "severity": "high"},
        {"category": "exposed_headers", "matched_text": "Access-Control-Expose-Headers", "severity": "medium"},
    ]


def test_detects_preflight_handling_and_max_age():
    result = detect_query_cors_requirements("CORS preflight requests should handle OPTIONS preflight and preflight max-age.")

    assert result["has_cors_requirements"] is True
    assert result["requirements"] == [
        {"category": "max_age", "matched_text": "preflight max-age", "severity": "low"},
        {"category": "preflight", "matched_text": "preflight requests", "severity": "high"},
    ]


def test_ignores_generic_origin_wording_without_browser_or_cross_origin_context():
    assert detect_query_cors_requirements("Explain the origin of source data for the analytics report.") == {
        "has_cors_requirements": False,
        "requirements": [],
    }
