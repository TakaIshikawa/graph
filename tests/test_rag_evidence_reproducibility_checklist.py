from __future__ import annotations

from graph.rag.evidence_reproducibility_checklist import build_evidence_reproducibility_checklist


def statuses(row):
    return {check["name"]: check["status"] for check in row["checks"]}


def test_reproducibility_checklist_detects_common_empirical_signals():
    checklist = build_evidence_reproducibility_checklist(
        [
            {
                "id": "study",
                "snippet": "Methods describe n=240 participants and a preregistered protocol.",
                "metadata": {"data": "Dataset and GitHub code availability statement"},
            }
        ]
    )

    assert checklist["evidence_count"] == 1
    assert statuses(checklist["items"][0]) == {
        "data_availability": "present",
        "methods_detail": "present",
        "code_availability": "present",
        "sample_size": "present",
        "preregistration": "present",
    }
    assert checklist["items"][0]["reproducibility_score"] == 1.0


def test_reproducibility_checklist_marks_missing_and_empty_unknown():
    weak = build_evidence_reproducibility_checklist([{"id": "note", "snippet": "Opinion summary."}])
    empty = build_evidence_reproducibility_checklist([])

    assert set(statuses(weak["items"][0]).values()) == {"missing"}
    assert weak["summary"]["missing"] == 5
    assert empty["warnings"] == ["no_evidence"]
    assert empty["summary"]["unknown"] == 5
