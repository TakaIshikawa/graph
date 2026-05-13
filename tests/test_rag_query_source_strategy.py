from __future__ import annotations

from dataclasses import dataclass

import pytest

from graph.rag import plan_query_source_strategy


@dataclass
class UnitStub:
    metadata: dict


@dataclass
class ResultStub:
    metadata: dict
    unit: UnitStub | None = None


def test_plan_query_source_strategy_maps_query_cues_to_source_types():
    payload = plan_query_source_strategy(
        "Compare papers, docs, issues, notes, bookmarks, and citations about vector search.",
        min_source_types=1,
    )

    assert payload["query_terms"] == [
        "compare",
        "papers",
        "docs",
        "issues",
        "notes",
        "bookmarks",
        "citations",
        "vector",
        "search",
    ]
    assert payload["requested_source_types"] == [
        "paper",
        "doc",
        "issue",
        "note",
        "bookmark",
        "citation",
    ]
    assert payload["observed_source_types"] == []
    assert payload["missing_source_types"] == [
        "bookmark",
        "citation",
        "doc",
        "issue",
        "note",
        "paper",
        "additional_source_type",
    ]
    assert payload["source_type_count"] == 0
    assert payload["needs_more_sources"] is True


def test_plan_query_source_strategy_extracts_observed_types_from_nested_results():
    payload = plan_query_source_strategy(
        "Use papers and docs with issue notes.",
        results=[
            {"source_type": "paper"},
            {"metadata": {"type": "documentation"}},
            ResultStub(metadata={"kind": "github_issue"}),
            {"unit": {"metadata": {"source_entity_type": "note"}}},
            ({"unit": UnitStub(metadata={"entity_type": "bookmark"})}, 0.8),
        ],
        min_source_types=3,
    )

    assert payload["requested_source_types"] == ["paper", "doc", "issue", "note"]
    assert payload["observed_source_types"] == ["bookmark", "doc", "issue", "note", "paper"]
    assert payload["missing_source_types"] == []
    assert payload["source_type_count"] == 5
    assert payload["needs_more_sources"] is False
    assert payload["recommendations"] == [
        "Current results satisfy requested source-type coverage."
    ]


def test_plan_query_source_strategy_reflects_required_types_and_minimum_sources():
    payload = plan_query_source_strategy(
        "Summarize launch blockers.",
        results=[{"metadata": {"source_entity_type": "issue"}}],
        required_source_types=["docs", "notes", "issue"],
        min_source_types=3,
    )

    assert payload["requested_source_types"] == ["doc", "note", "issue"]
    assert payload["observed_source_types"] == ["issue"]
    assert payload["missing_source_types"] == ["doc", "note", "additional_source_type"]
    assert payload["needs_more_sources"] is True
    assert payload["recommendations"] == [
        "Retrieve requested source types: doc, note.",
        "Add at least 2 more distinct source types.",
    ]


def test_plan_query_source_strategy_handles_no_requested_types_with_enough_sources():
    payload = plan_query_source_strategy(
        "Summarize semantic search rollout.",
        results=[
            {"source_type": "meeting_note"},
            {"metadata": {"source_type": "bookmark"}},
        ],
    )

    assert payload["requested_source_types"] == []
    assert payload["observed_source_types"] == ["bookmark", "meeting_note"]
    assert payload["missing_source_types"] == []
    assert payload["needs_more_sources"] is False
    assert payload["recommendations"] == [
        "Add explicit source-type constraints if the answer needs provenance diversity."
    ]


@pytest.mark.parametrize("query", ["", "   ", None])
def test_plan_query_source_strategy_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        plan_query_source_strategy(query)  # type: ignore[arg-type]


@pytest.mark.parametrize("min_source_types", [0, -1, True])
def test_plan_query_source_strategy_validates_min_source_types(min_source_types):
    with pytest.raises(ValueError, match="min_source_types"):
        plan_query_source_strategy("query", min_source_types=min_source_types)


@pytest.mark.parametrize("required_source_types", [[], ["docs", "docs"], [""]])
def test_plan_query_source_strategy_validates_and_deduplicates_required_types(required_source_types):
    if required_source_types == [""]:
        with pytest.raises(ValueError, match="required_source_types"):
            plan_query_source_strategy("query", required_source_types=required_source_types)
    else:
        payload = plan_query_source_strategy(
            "query",
            required_source_types=required_source_types,
            min_source_types=1,
        )
        assert payload["requested_source_types"] == (["doc"] if required_source_types else [])
