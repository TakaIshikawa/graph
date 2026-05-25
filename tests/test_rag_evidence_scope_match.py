from __future__ import annotations

from dataclasses import dataclass

from graph.rag.evidence_scope_match import score_evidence_scope_match


@dataclass
class Result:
    id: str
    content: str
    metadata: dict[str, str]


def test_scores_scope_match_across_dimensions_from_dicts_and_objects():
    evidence = [
        {"id": "a", "content": "US enterprise customer revenue in 2026", "metadata": {"metric": "revenue"}},
        Result("b", "Japan consumer latency", {"date": "2024"}),
    ]

    report = score_evidence_scope_match("US enterprise customer revenue 2026", evidence)

    assert report["query_scope"] == {
        "entity": ["customer"],
        "geography": ["us"],
        "metric": ["revenue"],
        "population": ["enterprise"],
        "time": ["2026"],
    }
    assert report["evidence"][0]["scope_score"] == 1.0
    assert report["evidence"][1]["missing_dimensions"] == ["entity", "geography", "metric", "population", "time"]


def test_empty_query_or_evidence_has_stable_outputs():
    assert score_evidence_scope_match("", []) == {"query_scope": {}, "evidence": []}
    report = score_evidence_scope_match("", [{"content": "anything"}])
    assert report["evidence"][0]["scope_score"] == 1.0
