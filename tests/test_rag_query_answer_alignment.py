from __future__ import annotations

from dataclasses import dataclass

from graph.rag.query_answer_alignment import check_query_answer_alignment


@dataclass
class Result:
    text: str
    metadata: dict | None = None


def test_query_answer_alignment_scores_missing_and_unsupported_terms():
    report = check_query_answer_alignment(
        "Explain Acme Cloud migration risks",
        "Acme Cloud has latency concerns and budget overruns.",
        [{"content": "Acme Cloud migration risks include latency concerns."}],
    )

    assert report["label"] == "partially-aligned"
    assert report["missing_query_terms"] == ["migration", "risks"]
    assert report["unsupported_answer_terms"] == ["budget", "overruns"]


def test_query_answer_alignment_accepts_tuple_object_results():
    report = check_query_answer_alignment(
        "Summarize Grid Alpha launch",
        "Grid Alpha launch is supported by Battery Pack evidence.",
        [(Result("Battery Pack evidence supports Grid Alpha launch."), 0.9)],
    )

    assert report["label"] == "aligned"
    assert report["missing_query_terms"] == []
    assert report["unsupported_answer_terms"] == []
