from graph.rag.query_cookie_security_requirement import detect_query_cookie_security_requirements


def test_detects_cookie_security_flags_and_samesite():
    result = detect_query_cookie_security_requirements("Cookies must set HttpOnly, Secure flag, and SameSite=Lax.")

    assert result["has_cookie_security_requirements"] is True
    assert [row["category"] for row in result["requirements"]] == ["httponly", "samesite", "secure"]


def test_detects_scope_expiration_and_signed_encrypted_cookies():
    result = detect_query_cookie_security_requirements("Configure cookie domain/path scope, Max-Age, signed cookies, and encrypted cookies.")

    assert [row["category"] for row in result["requirements"]] == ["expiration", "scope", "signed_encrypted"]


def test_avoids_unrelated_browser_cookie_mentions():
    assert detect_query_cookie_security_requirements("Explain why browsers store cookies for preferences.") == {
        "has_cookie_security_requirements": False,
        "requirements": [],
    }
