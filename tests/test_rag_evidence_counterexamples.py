from __future__ import annotations

from graph.rag.evidence_counterexamples import find_evidence_counterexamples


def test_finds_grouped_counterexample_passages_with_cue_types():
    report = find_evidence_counterexamples(
        [
            {
                "id": "r1",
                "content": "The treatment helped most patients. However, it did not improve sleep. Results were limited by a small sample.",
            }
        ],
        max_passages_per_result=2,
    )

    assert report["counterexample_count"] == 2
    assert report["results"][0]["result_id"] == "r1"
    assert {match["cue_type"] for match in report["results"][0]["matches"]} == {"conflicting_condition", "negative_finding"}


def test_passage_extraction_is_bounded():
    report = find_evidence_counterexamples(
        [{"id": "r1", "text": "Except for adults, it applies. However, children differ. No evidence supports infants."}],
        max_passages_per_result=1,
    )

    assert report["results"][0]["match_count"] == 1
