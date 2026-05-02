from __future__ import annotations

import pytest

from graph.rag import highlight_result_snippets


def test_highlight_result_snippets_matches_case_insensitively_preserving_casing():
    results = [
        {
            "id": "solar",
            "content": "Intro text. SOLAR storage improves planning.",
            "score": 0.9,
        }
    ]

    highlighted = highlight_result_snippets(results, "solar", snippet_chars=80)

    assert highlighted == [
        {
            "id": "solar",
            "content": "Intro text. SOLAR storage improves planning.",
            "score": 0.9,
            "snippet": "Intro text. <mark>SOLAR</mark> storage improves planning.",
            "highlighted_terms": ["solar"],
        }
    ]


def test_highlight_result_snippets_highlights_multiple_terms_deterministically():
    result = highlight_result_snippets(
        [
            {
                "id": "battery",
                "content": "Battery storage pairs with solar. Storage supports the grid.",
            }
        ],
        "storage solar storage",
        snippet_chars=120,
    )[0]

    assert result["snippet"] == (
        "Battery <mark>storage</mark> pairs with <mark>solar</mark>. "
        "<mark>Storage</mark> supports the grid."
    )
    assert result["highlighted_terms"] == ["storage", "solar"]


def test_highlight_result_snippets_supports_custom_content_key_and_markers():
    result = highlight_result_snippets(
        [{"id": "custom", "body": "Grid batteries reduce peak demand."}],
        "batteries demand",
        content_key="body",
        snippet_chars=80,
        marker_start="[[",
        marker_end="]]",
    )[0]

    assert result["snippet"] == "Grid [[batteries]] reduce peak [[demand]]."
    assert result["highlighted_terms"] == ["batteries", "demand"]


def test_highlight_result_snippets_uses_leading_snippet_without_matches():
    result = highlight_result_snippets(
        [{"id": "no-match", "content": "Battery market context without the term."}],
        "solar",
        snippet_chars=22,
    )[0]

    assert result["snippet"] == "Battery market context"
    assert result["highlighted_terms"] == []


def test_highlight_result_snippets_handles_missing_content():
    result = highlight_result_snippets([{"id": "missing"}], "solar")[0]

    assert result == {"id": "missing", "snippet": "", "highlighted_terms": []}


def test_highlight_result_snippets_does_not_mutate_input_results():
    original = {"id": "unit", "content": "Solar storage", "metadata": {"rank": 1}}
    before = dict(original)

    highlighted = highlight_result_snippets([original], "solar")

    assert original == before
    assert highlighted[0] is not original
    assert highlighted[0]["metadata"] is original["metadata"]
    assert highlighted[0]["snippet"] == "<mark>Solar</mark> storage"


@pytest.mark.parametrize("snippet_chars", [-1, "2", True])
def test_highlight_result_snippets_validates_snippet_chars(snippet_chars):
    with pytest.raises(ValueError, match="snippet_chars must be a non-negative integer"):
        highlight_result_snippets([], "solar", snippet_chars=snippet_chars)


def test_highlight_result_snippets_is_importable_from_graph_rag():
    from graph.rag import highlight_result_snippets as imported

    assert imported is highlight_result_snippets
