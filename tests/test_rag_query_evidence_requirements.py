from __future__ import annotations

import pytest

from graph.rag.query_evidence_requirements import classify_query_evidence_requirements


FLAGS = (
    "requires_recency",
    "requires_comparison",
    "requires_source_diversity",
    "requires_citations",
    "requires_quantitative_evidence",
    "requires_synthesis_steps",
)


def test_classify_query_evidence_requirements_returns_stable_flags_and_reasons():
    payload = classify_query_evidence_requirements(
        "Compare the latest adoption metrics from multiple sources and cite links."
    )

    assert list(payload) == [*FLAGS, "reasons", "normalized_query"]
    assert {flag: payload[flag] for flag in FLAGS} == {
        "requires_recency": True,
        "requires_comparison": True,
        "requires_source_diversity": True,
        "requires_citations": True,
        "requires_quantitative_evidence": True,
        "requires_synthesis_steps": False,
    }
    assert payload["reasons"] == {
        "requires_recency": ["cue:latest"],
        "requires_comparison": ["cue:compare"],
        "requires_source_diversity": ["cue:multiple-sources"],
        "requires_citations": ["cue:cite", "cue:sources", "cue:links"],
        "requires_quantitative_evidence": ["cue:metrics"],
        "requires_synthesis_steps": [],
    }


def test_classify_query_evidence_requirements_normalizes_case_and_whitespace():
    first = classify_query_evidence_requirements("  STEP-by-step   Plan for CURRENT rates  ")
    second = classify_query_evidence_requirements("step-by-step plan for current rates")

    assert first == second
    assert first["normalized_query"] == "step-by-step plan for current rates"
    assert first["requires_recency"] is True
    assert first["requires_quantitative_evidence"] is True
    assert first["requires_synthesis_steps"] is True


def test_classify_query_evidence_requirements_leaves_unmatched_flags_false():
    payload = classify_query_evidence_requirements("Explain semantic search")

    assert {flag: payload[flag] for flag in FLAGS} == dict.fromkeys(FLAGS, False)
    assert payload["reasons"] == {flag: [] for flag in FLAGS}


@pytest.mark.parametrize("query", ["", "   ", None, 42])
def test_classify_query_evidence_requirements_validates_query(query):
    with pytest.raises(ValueError, match="query must be a non-empty string"):
        classify_query_evidence_requirements(query)  # type: ignore[arg-type]
