from __future__ import annotations

from graph.rag.evidence_sampling_bias import detect_evidence_sampling_bias


def test_sampling_bias_groups_bias_types_by_evidence_id():
    result = detect_evidence_sampling_bias(
        [
            {"id": "e1", "text": "Convenience sample from an online survey; not representative."},
            {"id": "e2", "metadata": {"limitations": "Small sample with attrition and nonresponse."}},
        ]
    )

    assert result["biased_evidence"] == [
        {"evidence_id": "e1", "bias_types": ["convenience_sample", "online_survey", "not_representative"]},
        {"evidence_id": "e2", "bias_types": ["small_sample", "attrition", "nonresponse"]},
    ]
    assert result["bias_types"] == [
        "convenience_sample",
        "online_survey",
        "small_sample",
        "attrition",
        "nonresponse",
        "not_representative",
    ]
    assert result["evidence_ids_by_bias_type"]["online_survey"] == ["e1"]
    assert result["warnings"] == ["sampling_bias_cues_detected"]


def test_sampling_bias_returns_no_warnings_without_cues():
    result = detect_evidence_sampling_bias([{"id": "e1", "text": "National probability sample."}])

    assert result["biased_evidence"] == []
    assert result["warnings"] == []
    assert result["confidence"] == 0.0
