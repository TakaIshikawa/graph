from __future__ import annotations

from graph.rag.answer_source_role_labels import label_answer_source_roles


def test_labels_bracket_url_and_named_source_mentions():
    report = label_answer_source_roles(
        "According to CDC Report [1], the study found lower risk. "
        "However https://example.test/case is a counterexample. "
        "The Oxford Dictionary defines the term."
    )

    sources = {row["source"]: row["role"] for row in report["source_mentions"]}
    assert sources["[1]"] == "primary_evidence"
    assert sources["CDC Report"] == "primary_evidence"
    assert sources["https://example.test/case"] == "counterexample"
    assert sources["Oxford Dictionary"] == "definition"
    assert report["role_counts"]["primary_evidence"] == 2


def test_defaults_uncued_sources_to_background_context():
    report = label_answer_source_roles("From Example Source, the topic has a long history.")

    assert report["source_mentions"][0]["role"] == "background_context"
