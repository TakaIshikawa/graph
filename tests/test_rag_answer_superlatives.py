from __future__ import annotations

from graph.rag.answer_superlatives import detect_unsupported_superlatives


def test_flags_superlative_without_comparative_evidence():
    report = detect_unsupported_superlatives("This is the best option.", [{"id": "r1", "text": "This option has a feature list."}])

    assert report["total_superlatives"] == 1
    assert report["unsupported_superlatives"] == 1
    assert report["reason_counts"] == {"missing_comparative_or_ranking_evidence": 1}


def test_supported_ranking_evidence_lowers_unsupported_count():
    report = detect_unsupported_superlatives("This is the highest ranked option.", [{"id": "rank", "text": "The option is highest in the ranking benchmark."}])

    assert report["supported_superlatives"] == 1
    assert report["unsupported_superlatives"] == 0
    assert report["claims"][0]["matched_result_ids"] == ["rank"]


def test_empty_inputs_warn_without_raising():
    report = detect_unsupported_superlatives("", [])

    assert report["warnings"] == ["no_answer", "no_results"]
    assert report["claims"] == []
