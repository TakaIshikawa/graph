from __future__ import annotations

from dataclasses import dataclass

from graph.rag.result_evidence_method_mix import analyze_result_evidence_method_mix


@dataclass
class ResultStub:
    id: str
    title: str = ""
    content: str = ""
    metadata: dict | None = None
    tags: list[str] | None = None


def test_analyze_result_evidence_method_mix_classifies_mixed_methods():
    payload = analyze_result_evidence_method_mix(
        [
            {
                "id": "study",
                "title": "Peer-reviewed study of 500 participants",
                "content": "The dataset reports confidence intervals.",
            },
            {
                "id": "docs",
                "title": "API reference",
                "url": "https://docs.example.com/api",
            },
            {
                "id": "news",
                "source": "Reuters",
                "content": "The reporter said the company announced the change.",
            },
            {
                "id": "opinion",
                "title": "Opinion: my view of retrieval quality",
            },
            {
                "id": "forum",
                "domain": "stackoverflow.com",
                "tags": ["community", "q&a"],
            },
            {
                "id": "reference",
                "metadata": {"genre": "encyclopedia reference"},
            },
        ]
    )

    assert payload["method_counts"] == {
        "empirical_study": 1,
        "official_documentation": 1,
        "news_reporting": 1,
        "opinion": 1,
        "forum": 1,
        "reference": 1,
        "unknown": 0,
    }
    assert payload["dominant_method"] == "empirical_study"
    assert payload["diversity_score"] == 1.0
    assert [row["method"] for row in payload["per_result"]] == [
        "empirical_study",
        "official_documentation",
        "news_reporting",
        "opinion",
        "forum",
        "reference",
    ]


def test_analyze_result_evidence_method_mix_reports_single_method_dominance():
    payload = analyze_result_evidence_method_mix(
        [
            {"id": "a", "title": "Official documentation"},
            {"id": "b", "url": "https://developer.example.com/guide"},
            {"id": "c", "metadata": {"source_type": "manual", "kind": "docs"}},
        ]
    )

    assert payload["method_counts"]["official_documentation"] == 3
    assert payload["dominant_method"] == "official_documentation"
    assert payload["diversity_score"] == 0.0
    assert payload["method_share"]["official_documentation"] == 1.0


def test_analyze_result_evidence_method_mix_handles_unknown_inputs():
    payload = analyze_result_evidence_method_mix(
        [
            {"id": "empty", "title": "Notes"},
            ResultStub(id="object", title="Miscellaneous clipping", metadata={"folder": "inbox"}),
            None,
        ]
    )

    assert payload["total_results"] == 3
    assert payload["method_counts"]["unknown"] == 3
    assert payload["dominant_method"] == "unknown"
    assert payload["diversity_score"] == 0.0
    assert payload["per_result"] == [
        {"result_id": "empty", "method": "unknown", "cues": []},
        {"result_id": "object", "method": "unknown", "cues": []},
        {"result_id": "result-3", "method": "unknown", "cues": []},
    ]


def test_analyze_result_evidence_method_mix_empty_input_is_deterministic():
    assert analyze_result_evidence_method_mix([]) == {
        "total_results": 0,
        "method_counts": {
            "empirical_study": 0,
            "official_documentation": 0,
            "news_reporting": 0,
            "opinion": 0,
            "forum": 0,
            "reference": 0,
            "unknown": 0,
        },
        "method_share": {
            "empirical_study": 0.0,
            "official_documentation": 0.0,
            "news_reporting": 0.0,
            "opinion": 0.0,
            "forum": 0.0,
            "reference": 0.0,
            "unknown": 0.0,
        },
        "dominant_method": "unknown",
        "diversity_score": 0.0,
        "per_result": [],
    }
