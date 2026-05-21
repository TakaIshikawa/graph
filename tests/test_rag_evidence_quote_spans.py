from __future__ import annotations

import pytest

from graph.rag.evidence_quote_spans import extract_evidence_quote_spans


def test_evidence_quote_spans_prefers_concrete_sentences_and_offsets_are_original():
    text = "This is generic context. Acme Research measured 42 failures on 2026-02-01 using table 4. Another generic sentence."
    report = extract_evidence_quote_spans([{"id": "r1", "text": text}])
    span = report["results"][0]["spans"][0]

    assert span["text"] == "Acme Research measured 42 failures on 2026-02-01 using table 4."
    assert text[span["start"] : span["end"]].strip() == span["text"]
    assert set(span["cues"]) >= {"number", "date", "named_reference", "method_detail"}


def test_evidence_quote_spans_honors_max_spans_per_result_deterministically():
    text = "One measured 10 cases. Two measured 20 cases. Three measured 30 cases."
    report = extract_evidence_quote_spans([{"id": "r", "text": text}], max_spans_per_result=2)

    assert [span["text"] for span in report["results"][0]["spans"]] == [
        "One measured 10 cases.",
        "Two measured 20 cases.",
    ]


def test_evidence_quote_spans_reports_sparse_evidence_and_validates_limit():
    assert extract_evidence_quote_spans([{"id": "empty", "text": "General background."}])["warnings"] == ["sparse_evidence_spans"]
    with pytest.raises(ValueError):
        extract_evidence_quote_spans([], max_spans_per_result=-1)
