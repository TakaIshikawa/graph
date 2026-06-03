from graph.rag.query_api_pagination_requirement import detect_query_api_pagination_requirements


def test_detects_cursor_token_total_count_and_stable_ordering():
    result = detect_query_api_pagination_requirements(
        "For the list users API, require cursor-based pagination, next page tokens, total count, and stable ordering."
    )

    assert result["has_api_pagination_requirements"] is True
    assert result["requirements"] == [
        {"category": "cursor_pagination", "matched_text": "cursor-based pagination", "severity": "high"},
        {"category": "next_token", "matched_text": "next page tokens", "severity": "high"},
        {"category": "stable_ordering", "matched_text": "stable ordering", "severity": "high"},
        {"category": "total_count", "matched_text": "total count", "severity": "medium"},
    ]


def test_detects_offset_pagination_and_page_size_limits():
    result = detect_query_api_pagination_requirements(
        "Search endpoint results need limit/offset pagination with a max page size."
    )

    assert result["has_api_pagination_requirements"] is True
    assert result["requirements"] == [
        {"category": "offset_pagination", "matched_text": "limit/offset", "severity": "medium"},
        {"category": "page_size_limit", "matched_text": "max page size", "severity": "medium"},
    ]


def test_detects_export_page_size_limit():
    result = detect_query_api_pagination_requirements("Bulk export must document per-page limits and continuation tokens.")

    assert result["has_api_pagination_requirements"] is True
    assert result["requirements"] == [
        {"category": "next_token", "matched_text": "continuation tokens", "severity": "high"},
        {"category": "page_size_limit", "matched_text": "per-page limits", "severity": "medium"},
    ]


def test_ignores_generic_webpage_pagination_mentions():
    assert detect_query_api_pagination_requirements("Improve the webpage pagination controls in the page layout.") == {
        "has_api_pagination_requirements": False,
        "requirements": [],
    }
