from graph.rag.query_session_management_requirement import detect_query_session_management_requirements


def test_detects_session_timeout_and_revocation_requirements():
    result = detect_query_session_management_requirements(
        "For user sessions after login, require idle timeout, absolute session timeout, and revoke user sessions."
    )

    assert result["has_session_management_requirements"] is True
    assert result["requirements"] == [
        {"category": "absolute_timeout", "matched_text": "absolute session timeout", "severity": "high"},
        {"category": "idle_timeout", "matched_text": "idle timeout", "severity": "high"},
        {"category": "session_revocation", "matched_text": "revoke user sessions", "severity": "high"},
    ]


def test_detects_cookie_backed_sessions_and_concurrency():
    result = detect_query_session_management_requirements(
        "Authentication sessions use cookie-backed sessions, remember me, concurrent sessions limits, and log out all devices."
    )

    assert result["has_session_management_requirements"] is True
    assert result["requirements"] == [
        {"category": "concurrent_sessions", "matched_text": "concurrent sessions", "severity": "medium"},
        {"category": "device_logout", "matched_text": "log out all devices", "severity": "medium"},
        {"category": "remember_me", "matched_text": "remember me", "severity": "medium"},
        {"category": "secure_cookie", "matched_text": "cookie-backed sessions", "severity": "high"},
    ]


def test_ignores_unrelated_browser_session_language():
    assert detect_query_session_management_requirements("Restore the browser session tabs after a crash.") == {
        "has_session_management_requirements": False,
        "requirements": [],
    }
