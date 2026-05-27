from __future__ import annotations

from graph.rag.context_source_role_balance import analyze_context_source_role_balance


def test_context_source_role_balance_uses_explicit_metadata():
    report = analyze_context_source_role_balance([{"id": "a", "role": "primary"}, {"id": "b", "source_role": "data"}])

    assert report["counts"]["primary"] == 1
    assert report["counts"]["data"] == 1
    assert report["items"] == [{"id": "a", "role": "primary"}, {"id": "b", "role": "data"}]


def test_context_source_role_balance_falls_back_to_text_hints():
    report = analyze_context_source_role_balance([{"id": "r", "title": "Systematic review and synthesis"}, {"id": "o", "text": "Editorial opinion"}])

    assert report["items"] == [{"id": "r", "role": "secondary"}, {"id": "o", "role": "opinion"}]


def test_context_source_role_balance_warns_for_missing_primary_and_data():
    report = analyze_context_source_role_balance(
        [{"title": "Background explainer"}, {"title": "Review article"}, {"title": "Commentary"}]
    )

    assert report["warnings"] == ["missing_primary_sources", "missing_data_sources"]
