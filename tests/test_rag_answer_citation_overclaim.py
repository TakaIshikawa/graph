from __future__ import annotations

from graph.rag import audit_answer_citation_overclaims


def test_answer_citation_overclaim_counts_uncited_strong_claims():
    report = audit_answer_citation_overclaims("This always works. It is the best option [1]. A caveat remains.")

    assert report["sentence_count"] == 3
    assert report["strong_claim_count"] == 2
    assert report["uncited_strong_claim_count"] == 1
    assert report["samples"] == [
        {"sentence_index": 1, "cue": "always", "sentence": "This always works.", "has_citation": False},
        {"sentence_index": 2, "cue": "best", "sentence": "It is the best option [1].", "has_citation": True},
    ]
