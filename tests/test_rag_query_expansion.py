from __future__ import annotations

import pytest

from graph.rag import suggest_query_expansion_terms


def test_query_expansion_supports_nested_unit_payloads_and_excludes_query_terms():
    payload = suggest_query_expansion_terms(
        [
            {
                "unit": {
                    "id": "u1",
                    "title": "Vector search recall",
                    "content": "Embeddings improve semantic recall for sparse notes.",
                    "tags": ["semantic-search", "retrieval"],
                    "metadata": {"keywords": ["nearest neighbors", {"term": "hybrid ranking"}]},
                }
            },
            {
                "unit": {
                    "id": "u2",
                    "title": "Hybrid retrieval",
                    "content": "Hybrid search combines sparse signals and embeddings.",
                    "metadata": {"metadata_keywords": ["semantic expansion"]},
                }
            },
        ],
        "semantic search recall",
        max_terms=6,
    )

    assert payload["query_terms"] == ["recall", "search", "semantic"]
    assert [item["term"] for item in payload["expansion_terms"]] == [
        "hybrid",
        "embeddings",
        "retrieval",
        "sparse",
        "combines",
        "expansion",
    ]
    assert payload["supporting_result_ids"]["hybrid"] == ["u1", "u2"]
    assert payload["supporting_result_ids"]["embeddings"] == ["u1", "u2"]


def test_query_expansion_extracts_tags_and_metadata_keywords():
    payload = suggest_query_expansion_terms(
        [
            {
                "id": "a",
                "title": "Short note",
                "tags": ["citation graphs"],
                "metadata": {"keywords": [{"keyword": "source audit"}, "coverage map"]},
            }
        ],
        "note",
    )

    assert [item["term"] for item in payload["expansion_terms"]] == [
        "audit",
        "citation",
        "coverage",
        "graphs",
        "map",
        "short",
        "source",
    ]
    assert payload["supporting_result_ids"]["audit"] == ["a"]


def test_query_expansion_orders_by_frequency_result_count_then_term():
    payload = suggest_query_expansion_terms(
        [
            {"id": "r1", "content": "alpha beta beta delta"},
            {"id": "r2", "content": "alpha gamma gamma delta"},
            {"id": "r3", "content": "alpha"},
        ],
        "query",
    )

    assert [item["term"] for item in payload["expansion_terms"][:4]] == [
        "alpha",
        "delta",
        "beta",
        "gamma",
    ]
    assert payload["expansion_terms"][0]["frequency"] == 3
    assert payload["expansion_terms"][0]["supporting_result_ids"] == ["r1", "r2", "r3"]


def test_query_expansion_limits_terms():
    payload = suggest_query_expansion_terms(
        [{"id": "r1", "content": "alpha beta gamma"}],
        "query",
        max_terms=2,
    )

    assert [item["term"] for item in payload["expansion_terms"]] == ["alpha", "beta"]


@pytest.mark.parametrize("max_terms", [0, -1, True, "3"])
def test_query_expansion_validates_max_terms(max_terms):
    with pytest.raises(ValueError, match="max_terms must be a positive integer"):
        suggest_query_expansion_terms([], "query", max_terms=max_terms)
