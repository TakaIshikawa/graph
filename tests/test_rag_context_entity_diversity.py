from __future__ import annotations

from graph.rag.context_entity_diversity import summarize_context_entity_diversity


def test_counts_metadata_arrays_and_scalars():
    result = summarize_context_entity_diversity([{"metadata": {"entities": ["Acme", "Beta"], "authors": "Acme"}}])

    assert result["entity_counts"]["Acme"] == 2
    assert result["entity_counts"]["Beta"] == 1


def test_includes_conservative_capitalized_content_cues():
    result = summarize_context_entity_diversity([{"content": "OpenAI released guidance. The policy mentions New York."}])

    assert result["entity_counts"]["OpenAI"] == 1
    assert result["entity_counts"]["New York"] == 1


def test_top_entities_and_ratios_are_deterministic():
    result = summarize_context_entity_diversity([{"entities": ["Beta", "Acme", "Acme"]}, {"entities": ["Beta"]}, {"entities": ["Solo"]}])

    assert result["top_entities"][:2] == [{"entity": "Acme", "count": 2}, {"entity": "Beta", "count": 2}]
    assert result["singleton_count"] == 1
    assert result["repeated_entity_ratio"] == 0.8
