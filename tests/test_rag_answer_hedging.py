from __future__ import annotations

from graph.rag.answer_hedging import audit_answer_hedging


def test_answer_hedging_flags_absolute_certainty_cues():
    report = audit_answer_hedging("This always works and proves the claim. It never fails.")

    assert report["counts"]["unsupported_certainty"] == 2
    assert report["balance_bucket"] == "overconfident"
    assert {record["cue"] for record in report["records"] if record["kind"] == "unsupported_certainty"} >= {"always", "never"}


def test_answer_hedging_flags_dense_uncertainty_cues():
    report = audit_answer_hedging("Maybe this might possibly work. The result is unclear and could change.")

    assert report["counts"]["excessive_uncertainty"] == 2
    assert report["balance_bucket"] == "over_hedged"


def test_answer_hedging_evidence_phrasing_reduces_certainty_severity():
    report = audit_answer_hedging("According to the cited study, this always improved latency.")

    certainty = [record for record in report["records"] if record["kind"] == "unsupported_certainty"][0]
    assert certainty["severity"] == "low"
    assert report["counts"]["evidence_reference"] == 1
    assert report["balance_bucket"] == "evidence_balanced"
