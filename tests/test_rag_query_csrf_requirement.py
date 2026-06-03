from graph.rag.query_csrf_requirement import detect_query_csrf_requirements


def test_detects_csrf_token_and_samesite_requirements():
    result = detect_query_csrf_requirements(
        "Browser web forms need CSRF tokens, SameSite cookies, and origin checks for session requests."
    )

    assert result["has_csrf_requirements"] is True
    assert result["requirements"] == [
        {"category": "csrf_token", "matched_text": "CSRF tokens", "severity": "high"},
        {"category": "origin_check", "matched_text": "origin checks", "severity": "high"},
        {"category": "same_site_cookie", "matched_text": "SameSite cookies", "severity": "high"},
    ]


def test_detects_unsafe_methods_double_submit_and_state_parameter():
    result = detect_query_csrf_requirements(
        "Protect POST requests with CSRF, support double-submit cookies, and validate the OAuth state parameter."
    )

    assert result["has_csrf_requirements"] is True
    assert result["requirements"] == [
        {"category": "double_submit_cookie", "matched_text": "double-submit cookies", "severity": "medium"},
        {"category": "state_parameter", "matched_text": "OAuth state parameter", "severity": "medium"},
        {"category": "unsafe_methods", "matched_text": "POST requests with CSRF", "severity": "high"},
    ]


def test_ignores_unrelated_state_management_queries():
    assert detect_query_csrf_requirements("Explain state management patterns for frontend reducers.") == {
        "has_csrf_requirements": False,
        "requirements": [],
    }
