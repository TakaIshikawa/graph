from graph.rag.query_browser_compatibility_requirement import detect_query_browser_compatibility_requirement


def test_detects_named_desktop_and_mobile_browsers():
    result = detect_query_browser_compatibility_requirement("Support Chrome, Firefox, Safari, Edge, and mobile browsers.")

    assert result["requires_browser_compatibility"] is True
    assert result["browsers"] == ["chrome", "edge", "firefox", "mobile_browser", "safari"]
    assert result["legacy_support_required"] is False


def test_version_and_legacy_cues_require_legacy_support():
    result = detect_query_browser_compatibility_requirement("Need legacy browsers including IE11 and Chrome <= 90.")

    assert result["legacy_support_required"] is True
    assert result["requires_browser_compatibility"] is True


def test_no_browser_cues_returns_empty_browser_list():
    assert detect_query_browser_compatibility_requirement("Need Linux support.") == {
        "requires_browser_compatibility": False,
        "browsers": [],
        "legacy_support_required": False,
    }
