from __future__ import annotations

from graph.rag.query_specificity import analyze_query_specificity


def test_analyze_query_specificity_classifies_one_word_query_as_broad():
    payload = analyze_query_specificity("energy")

    assert payload == {
        "specificity": "broad",
        "score": 0.035,
        "signals": {
            "token_count": 1,
            "quoted_phrase_count": 0,
            "operator_count": 0,
            "entity_like_count": 0,
            "date_count": 0,
            "concrete_id_count": 0,
            "has_url": False,
        },
        "suggested_refinements": [
            "add more topic terms",
            "add an exact phrase",
            "add a date or time range",
            "add a search operator such as site: or filetype:",
        ],
    }


def test_analyze_query_specificity_classifies_entity_rich_dated_query_as_specific():
    payload = analyze_query_specificity(
        '"OpenAI Responses API" pricing changes 2026 site:openai.com FILE-123'
    )

    assert payload["specificity"] == "specific"
    assert payload["score"] == 1.0
    assert payload["signals"] == {
        "token_count": 8,
        "quoted_phrase_count": 1,
        "operator_count": 2,
        "entity_like_count": 3,
        "date_count": 1,
        "concrete_id_count": 1,
        "has_url": False,
    }
    assert payload["suggested_refinements"] == []


def test_analyze_query_specificity_classifies_quoted_or_operator_query_as_focused():
    payload = analyze_query_specificity('"hybrid search" before:2025')

    assert payload["specificity"] == "focused"
    assert payload["score"] == 0.415
    assert payload["signals"]["quoted_phrase_count"] == 1
    assert payload["signals"]["operator_count"] == 1
    assert payload["signals"]["date_count"] == 1


def test_analyze_query_specificity_handles_empty_and_whitespace_queries():
    assert analyze_query_specificity("") == analyze_query_specificity("   ")
    assert analyze_query_specificity(None) == {
        "specificity": "broad",
        "score": 0.0,
        "signals": {
            "token_count": 0,
            "quoted_phrase_count": 0,
            "operator_count": 0,
            "entity_like_count": 0,
            "date_count": 0,
            "concrete_id_count": 0,
            "has_url": False,
        },
        "suggested_refinements": [
            "add more topic terms",
            "add an exact phrase",
            "add a date or time range",
            "add a search operator such as site: or filetype:",
        ],
    }


def test_analyze_query_specificity_counts_urls_as_concrete_identifiers():
    payload = analyze_query_specificity("compare https://example.com/report with NASA 2024")

    assert payload["specificity"] == "specific"
    assert payload["signals"]["has_url"] is True
    assert payload["signals"]["concrete_id_count"] == 1
