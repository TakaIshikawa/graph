from __future__ import annotations

from graph.rag.evidence_paywall_risk import summarize_evidence_paywall_risk


class EvidenceObject:
    def __init__(self) -> None:
        self.id = "obj-1"
        self.text = "Abstract only; purchase access for the complete paper."


def test_evidence_paywall_risk_flags_text_and_metadata_cues():
    report = summarize_evidence_paywall_risk(
        [
            "Subscription required to read this report.",
            {"id": "m1", "metadata": {"access": "sign in required"}},
            {"id": "open", "text": "Open access full text."},
        ]
    )

    assert report["total_items"] == 3
    assert report["paywall_risk_count"] == 2
    assert report["risk_items"] == [
        {"id": "result-1", "cues": ["subscription_required"]},
        {"id": "m1", "cues": ["sign_in_required"]},
    ]
    assert report["risk_ratio"] == 0.6667


def test_evidence_paywall_risk_supports_object_records_and_counts_cues():
    report = summarize_evidence_paywall_risk([EvidenceObject()])

    assert report["risk_items"] == [{"id": "obj-1", "cues": ["abstract_only", "purchase_access"]}]
    assert report["cue_counts"] == [{"cue": "abstract_only", "count": 1}, {"cue": "purchase_access", "count": 1}]
