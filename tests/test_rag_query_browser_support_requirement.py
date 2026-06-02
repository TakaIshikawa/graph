import pytest

from graph.rag.query_browser_support_requirement import detect_query_browser_support_requirement


def test_detects_browser_versions_and_support_cues():
    result = detect_query_browser_support_requirement(
        "Need a browser support compatibility matrix for Chrome 120, Safari 17, Firefox ESR, Chromium, WebKit, and IE11."
    )

    assert result["requires_browser_support"] is True
    assert result["cue_categories"] == [
        "browser_family",
        "version_floor",
        "legacy_browser",
        "rendering_engine",
        "compatibility_matrix",
    ]
    assert result["browser_versions"] == ["Chrome 120", "Safari 17", "Firefox ESR", "Chromium", "WebKit", "IE11"]


def test_non_browser_query_returns_empty_lists():
    assert detect_query_browser_support_requirement("Need Linux deployment evidence.") == {
        "requires_browser_support": False,
        "cue_categories": [],
        "browser_versions": [],
    }


@pytest.mark.parametrize("query", ["", "   ", None, 123])
def test_invalid_query_raises_value_error(query):
    with pytest.raises(ValueError):
        detect_query_browser_support_requirement(query)  # type: ignore[arg-type]
