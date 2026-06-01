from __future__ import annotations

from graph.store.source_graphql_hint_summary import summarize_source_graphql_hints


def test_source_graphql_hints_count_endpoint_and_textual_hints():
    summary = summarize_source_graphql_hints(
        [
            {"id": "s1", "url": "https://api.example.test/graphql"},
            {"id": "s2", "metadata": {"description": "GraphQL partner API"}},
        ]
    )

    assert summary["total_sources"] == 2
    assert summary["graphql_source_count"] == 2
    assert summary["samples"][0]["hint_type"] == "graphql"


def test_source_graphql_hints_count_operations_and_special_features():
    summary = summarize_source_graphql_hints(
        [
            {"id": "s1", "metadata": {"query": "query GetUser { user { id } }"}},
            {"id": "s2", "metadata": {"operation": "mutation UpdateUser"}},
            {"id": "s3", "metadata": {"description": "subscription stream with introspection and persisted query support"}},
        ]
    )

    assert summary["graphql_source_count"] == 3
    assert summary["operation_counts"] == {"mutation": 1, "query": 2, "subscription": 1}
    assert summary["introspection_hint_count"] == 1
    assert summary["persisted_query_count"] == 1


def test_source_graphql_hints_ignore_non_graphql_api_sources():
    assert summarize_source_graphql_hints([{"id": "s1", "url": "https://api.example.test/rest/users"}]) == {
        "total_sources": 1,
        "graphql_source_count": 0,
        "operation_counts": {},
        "introspection_hint_count": 0,
        "persisted_query_count": 0,
        "samples": [],
    }
